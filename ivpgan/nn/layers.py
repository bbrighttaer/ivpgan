# Author: bbrighttaer
# Project: ivpgan
# Date: 5/23/19
# Time: 1:39 PM
# File: layers.py


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import ABC

import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_max

from ivpgan.utils.math import segment_sum


def _proc_segment_ids(data, segment_ids):
    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"
    # segment_ids is a 1-D tensor repeat it to have the same shape as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long()
        s = check_cuda(s)
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])
    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"
    return segment_ids


def check_cuda(tensor):
    from ivpgan import cuda
    if cuda:
        return tensor.cuda()
    else:
        return tensor


class Layer(nn.Module, ABC):
    """
    Base class for creating layers
    """

    def __init__(self, activation=None):
        super(Layer, self).__init__()

        if activation and isinstance(activation, str):
            from ivpgan.nn.models import NonsatActivation
            self.activation = {'relu': nn.ReLU(),
                               'leaky_relu': nn.LeakyReLU(.2),
                               'sigmoid': nn.Sigmoid(),
                               'tanh': nn.Tanh(),
                               'softmax': nn.Softmax(),
                               'elu': nn.ELU(),
                               'nonsat': NonsatActivation()}.get(activation.lower(), nn.ReLU())
        else:
            self.activation = activation


class WeaveLayer(Layer):
    """
    PyTorch implementation of https://arxiv.org/pdf/1603.00856 guided by DeepChem's TF implementation.
    """

    def __init__(self, n_atom_input_feat=75, n_pair_input_feat=14, n_atom_output_feat=50, n_pair_output_feat=50,
                 n_hidden_AA=50, n_hidden_PA=50, n_hidden_AP=50, n_hidden_PP=50, update_pair=True, activation='relu'):
        """
        Parameters
        ----------
        n_atom_input_feat: int, optional
          Number of features for each atom in input.
        n_pair_input_feat: int, optional
          Number of features for each pair of atoms in input.
        n_atom_output_feat: int, optional
          Number of features for each atom in output.
        n_pair_output_feat: int, optional
          Number of features for each pair of atoms in output.
        update_pair: bool, optional
          Whether to calculate for pair features,
          could be turned off for last layer
        init: str, optional
          Weight initialization for filters.
        activation: str, optional
          Activation function applied
        """
        super(WeaveLayer, self).__init__(activation)
        # self.init = init  # Set weight initialization
        self.update_pair = update_pair  # last weave layer does not need to update
        self.n_hidden_AA = n_hidden_AA
        self.n_hidden_PA = n_hidden_PA
        self.n_hidden_AP = n_hidden_AP
        self.n_hidden_PP = n_hidden_PP
        self.n_hidden_A = n_hidden_AA + n_hidden_PA
        self.n_hidden_P = n_hidden_AP + n_hidden_PP

        self.n_atom_input_feat = n_atom_input_feat
        self.n_pair_input_feat = n_pair_input_feat
        self.n_atom_output_feat = n_atom_output_feat
        self.n_pair_output_feat = n_pair_output_feat

        # Trainable parameters
        self.linear_AA = Linear(self.n_atom_input_feat, self.n_hidden_AA, activation)
        self.linear_PA = Linear(self.n_pair_input_feat, self.n_hidden_PA, activation)
        self.linear_A = Linear(self.n_hidden_A, self.n_atom_output_feat, activation)
        if self.update_pair:
            self.linear_AP = Linear(self.n_atom_input_feat * 2, self.n_hidden_AP, activation)
            self.linear_PP = Linear(self.n_pair_input_feat, self.n_hidden_PP, activation)
            self.linear_P = Linear(self.n_hidden_P, self.n_pair_output_feat, activation)

    def forward(self, input_data):
        """
        Performs weave convolution on the input data.

        :param input_data: Must be an iterable [atom features, pair features, pair split, atom atom-pair]
        :return: (Atom features, Pair features)
        """
        atom_features, pair_features, pair_split, atom_split, atom_to_pair = input_data

        AA = self.linear_AA(atom_features)
        AA = self.activation(AA)
        PA = self.linear_PA(pair_features)
        PA = self.activation(PA)
        PA = segment_sum(data=PA, segment_ids=pair_split)
        A = self.linear_A(torch.cat([AA, PA], dim=1))
        A = self.activation(A)

        if self.update_pair:
            atom_to_pair = atom_to_pair
            AP_ij = self.linear_AP(atom_features[atom_to_pair].view(-1, 2 * self.n_atom_input_feat))
            AP_ji = self.linear_AP(atom_features[atom_to_pair.flip([1])].view(-1, 2 * self.n_atom_input_feat))
            if self.activation:
                AP_ij = self.activation(AP_ij)
                AP_ji = self.activation(AP_ji)

            PP = self.linear_PP(pair_features)
            PP = self.activation(PP) if self.activation else PP
            P = self.linear_P(torch.cat([AP_ij + AP_ji, PP], dim=1))
            P = self.activation(P) if self.activation else P
        else:
            P = pair_features
        return A, P


class WeaveGather(Layer):
    """
    Final layer of weave convolution.
    """

    def __init__(self, conv_out_depth, n_depth=128, gaussian_expand=False, activation='tanh', epsilon=1e-7):
        """
        Parameters
        ----------
        conv_out_depth: int, optional
          atom convolution depth
        gaussian_expand: boolean. optional
          Whether to expand each dimension of atomic features by gaussian histogram
        activation: str, optional
          Activation function applied

        """
        super(WeaveGather, self).__init__(activation)
        self.n_depth = n_depth
        self.conv_out_depth = conv_out_depth
        self.gaussian_expand = gaussian_expand
        self.epsilon = epsilon

        # Fuzzy membership functions
        self.gaussian_memberships = [(-1.645, 0.283), (-1.080, 0.170), (-0.739, 0.134),
                                     (-0.468, 0.118), (-0.228, 0.114), (0., 0.114),
                                     (0.228, 0.114), (0.468, 0.118), (0.739, 0.134),
                                     (1.080, 0.170), (1.645, 0.283)]
        self.linear = Linear(self.conv_out_depth * len(self.gaussian_memberships), self.n_depth, activation)

    def forward(self, input_data):
        outputs, atom_split = input_data

        if self.gaussian_expand:
            outputs = self.gaussian_histogram(outputs)
        output_molecules = segment_sum(outputs, atom_split)

        if self.gaussian_expand:
            output_molecules = self.linear(output_molecules)
            output_molecules = self.activation(output_molecules) if self.activation else output_molecules
        return output_molecules

    def gaussian_histogram(self, x):
        dist = [torch.distributions.normal.Normal(torch.tensor(m), torch.tensor(s))
                for m, s in self.gaussian_memberships]
        dist_max = [dist[i].log_prob(self.gaussian_memberships[i][0]) for i in range(len(self.gaussian_memberships))]
        outputs = [dist[i].log_prob(x) / dist_max[i] for i in range(len(self.gaussian_memberships))]
        outputs = torch.stack(outputs, dim=2)
        outputs = torch.add(outputs, torch.tensor(self.epsilon)) / torch.add(torch.sum(outputs, dim=2, keepdim=True),
                                                                             self.epsilon)
        outputs = outputs.view(-1, self.conv_out_depth * len(self.gaussian_memberships))
        return outputs


class WeaveDropout(Layer):

    def __init__(self, prob):
        """
        Creates a dropout layer for weave convolution.
        """
        super(WeaveDropout, self).__init__()
        self.atom_dropout = nn.Dropout(prob)
        self.pair_dropout = nn.Dropout(prob)

    def forward(self, atom_input, pair_input):
        return self.atom_dropout(atom_input), self.pair_dropout(pair_input)


class WeaveBatchNorm(Layer):

    def __init__(self, atom_dim, pair_dim, activation=None):
        super(WeaveBatchNorm, self).__init__(activation)
        self.a_batchnorm = nn.BatchNorm1d(atom_dim)
        self.p_batchnorm = nn.BatchNorm1d(pair_dim)

    def forward(self, atom_input, pair_input):
        return self.a_batchnorm(atom_input), self.p_batchnorm(pair_input)


class GraphConvLayer(Layer):
    def __init__(self, in_dim, out_dim, min_deg=0, max_deg=10, activation='relu'):
        super().__init__(activation)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.min_degree = min_deg
        self.max_degree = max_deg
        self.num_deg = 2 * max_deg + (1 - min_deg)

        # parameters
        self.linear_layers = nn.ModuleList()
        self._create_variables(self.in_dim)

    def _create_variables(self, in_channels):
        for i in range(self.num_deg):
            sub_lyr = nn.Linear(in_channels, self.out_dim)
            self.linear_layers.append(sub_lyr)

    def _sum_neighbors(self, atoms, deg_adj_lists):
        """Store the summed atoms by degree"""
        deg_summed = self.max_degree * [None]

        # TODO(bbrighttaer): parallelize
        for deg in range(1, self.max_degree + 1):
            idx = deg_adj_lists[deg - 1]
            gathered_atoms = atoms[idx.long()]
            # Sum along neighbors as well as self, and store
            summed_atoms = torch.sum(gathered_atoms, dim=1)
            deg_summed[deg - 1] = summed_atoms
        return deg_summed

    def forward(self, input_data):
        # in_layers = [atom_features, deg_slice, membership, deg_adj_list placeholders...]
        atom_features = input_data[0]

        # Graph topology info
        deg_slice = input_data[1]
        deg_adj_lists = input_data[3:]

        layers = iter(self.linear_layers)

        # Sum all neighbors using adjacency matrix
        deg_summed = self._sum_neighbors(atom_features, deg_adj_lists)

        # Get collection of modified atom features
        new_rel_atoms_collection = (self.max_degree + 1 - self.min_degree) * [None]

        # TODO(bbrighttaer): parallelize
        for deg in range(1, self.max_degree + 1):
            # Obtain relevant atoms for this degree
            rel_atoms = deg_summed[deg - 1]

            # Get self atoms
            begin = deg_slice[deg - self.min_degree, 0]
            size = deg_slice[deg - self.min_degree, 1]
            self_atoms = atom_features[begin: (begin + size), :]

            # Apply hidden affine to relevant atoms and append
            rel_out = next(layers)(rel_atoms.float())
            self_out = next(layers)(self_atoms.float())
            out = rel_out + self_out

            new_rel_atoms_collection[deg - self.min_degree] = out

        # Determine the min_deg=0 case
        if self.min_degree == 0:
            deg = 0

            begin = deg_slice[deg - self.min_degree, 0]
            size = deg_slice[deg - self.min_degree, 1]
            self_atoms = atom_features[begin:(begin + size), :]

            # Only use the self layer
            out = next(layers)(self_atoms.float())

            new_rel_atoms_collection[deg - self.min_degree] = out

        # Combine all atoms back into the list
        atom_features = torch.cat(new_rel_atoms_collection, dim=0)

        if self.activation is not None:
            atom_features = self.activation(atom_features)

        return atom_features


class GraphPool(Layer):
    def __init__(self, min_degree=0, max_degree=10):
        super(GraphPool, self).__init__()
        self.min_degree = min_degree
        self.max_degree = max_degree

    def forward(self, input_data):
        atom_features = input_data[0]
        deg_slice = input_data[1]
        deg_adj_lists = input_data[3:]

        deg_maxed = (self.max_degree + 1 - self.min_degree) * [None]

        # TODO(bbrighttaer): parallelize
        for deg in range(1, self.max_degree + 1):
            # Get self atoms
            begin = deg_slice[deg - self.min_degree, 0]
            size = deg_slice[deg - self.min_degree, 1]
            self_atoms = atom_features[begin:(begin + size), :]

            # Expand dims
            self_atoms = torch.unsqueeze(self_atoms, 1)

            gathered_atoms = atom_features[deg_adj_lists[deg - 1].long()]
            gathered_atoms = torch.cat([self_atoms, gathered_atoms], dim=1)

            if gathered_atoms.nelement() != 0:
                maxed_atoms = torch.max(gathered_atoms, 1)[0]  # info: [0] data [1] indices
                deg_maxed[deg - self.min_degree] = maxed_atoms

        if self.min_degree == 0:
            begin = deg_slice[0, 0]
            size = deg_slice[0, 1]
            self_atoms = atom_features[begin:(begin + size), :]
            deg_maxed[0] = self_atoms

        # Eliminate empty lists before concatenation
        deg_maxed = [d for d in deg_maxed if isinstance(d, torch.Tensor)]
        tensor_list = []
        for tensor in deg_maxed:
            if tensor.nelement() != 0:
                tensor_list.append(tensor.float())
        out_tensor = torch.cat(tensor_list, dim=0)
        return out_tensor


class GraphGather(Layer):

    def __init__(self, activation='tanh'):
        super(GraphGather, self).__init__(activation)

    def forward(self, input_data, batch_size):
        atom_features = input_data[0]

        # Graph topology
        membership = input_data[2]

        assert batch_size > 1, "Graph gather requires batches larger than 1"

        segment_ids = _proc_segment_ids(atom_features, membership)
        sparse_reps = scatter_add(atom_features, segment_ids, dim=0, fill_value=0)
        max_reps, _ = scatter_max(atom_features, segment_ids, dim=0, fill_value=0)
        mol_features = torch.cat([sparse_reps, max_reps], dim=1)

        if self.activation:
            mol_features = self.activation(mol_features) if self.activation else mol_features
        return mol_features


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, activation_name=None):
        """
        A custom linear module to facilitate weight initialization w.r.t. the activation function used after (if any)

        Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        activation: The label of the activation function used (if any)
        """
        self.activation_name = activation_name
        super(Linear, self).__init__(in_features, out_features, bias)


class Conv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', activation_name=None):
        """
        A custom Conv1d module to facilitate weight initialization w.r.t. the activation function used after (if any)

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param padding_mode:
        :param activation:
        """
        self.activation_name = activation_name
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                     padding_mode)


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', activation_name=None):
        """
        A custom Conv2d module to facilitate weight initialization w.r.t. the activation function used after (if any)

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param padding_mode:
        :param activation_name:
        """
        self.activation_name = activation_name
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                     padding_mode)


class Reshape(nn.Module):
    """
    Reshapes a tensor using the given shape. The batch size is the first argument and it's set by default.
    Thus, the passed shape does not include the batch size.
    """

    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        """
        Assumes that the shape of x has the structure: (batch_size, ...)
        :param x:
        :return:
        """
        return x.view(x.size()[0], *self.shape)


class Flatten(nn.Module):
    """
    Flattens each sample (e.g. outputs of a Conv model) of the batch
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Assumes that the shape of x has the structure: (batch_size, ...)
        :param x:
        :return:
        """
        return x.view(x.size()[0], -1)


class CIV(nn.Module):
    """Combined Input Vector module.
    It's basically a wrapper for torch.cat to enable its inclusion in nn.Sequential objects.
    """

    def __init__(self, dim):
        super(CIV, self).__init__()
        self.dim = dim

    def forward(self, input):
        combined = torch.cat(input, dim=self.dim)
        return combined


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(dim=1)
