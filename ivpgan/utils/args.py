# Author: bbrighttaer
# Project: ivpgan
# Date: 5/29/19
# Time: 4:34 PM
# File: args.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import ABC
from torch.nn import ReLU, Sigmoid, Tanh, ELU, Softmax, LeakyReLU


# Helper classes for packaging arguments to torch.nn modules.
from ivpgan.nn.models import NonsatActivation


class Args(ABC):
    def __init__(self, activation, batch_norm, dropout):
        """
        Parent class for wrapping arguments.

        :param activation: Activation function.
        :param batch_norm: Whether to add batch normalization after affine combination.
        :param dropout: The drop out probability. Pass -1 to disable the use of dropout.
        """
        self._args = None
        if activation:
            activation = {'relu': ReLU(),
                          'leaky_relu': LeakyReLU(.2),
                          'sigmoid': Sigmoid(),
                          'tanh': Tanh(),
                          'softmax': Softmax(dim=1),
                          'elu': ELU(),
                          'nonsat': NonsatActivation()}.get(activation.lower(), ReLU())
        self._activation = activation
        self._batch_norm = batch_norm
        self._dropout = dropout

    def __getitem__(self, i):
        return self._args[i]

    @property
    def args(self):
        return self._args

    @property
    def activation_function(self):
        return self._activation

    @property
    def use_batch_norm(self):
        return self._batch_norm

    @property
    def dropout(self):
        return self._dropout


class ConvArgs(Args):
    """Convolution arguments"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', activation=None, batch_norm=False, dropout=-1, pooling=None, conv_type='1D'):
        super(ConvArgs, self).__init__(activation, batch_norm, dropout)
        self.pooling = pooling
        self.conv_type = conv_type
        self._args = (in_channels, out_channels, kernel_size,
                      stride, padding, dilation, groups, bias, padding_mode)


class PoolingArg:
    """Pooling layer arguments wrapper. To be used as an argument to ConvArgs"""

    def __init__(self, ptype, **kwargs):
        """

        :param ptype: Pooling type e.g. max_pool or avg_pool
        :param kwargs: arguments for creating a pooling layer
        """
        self.ptype = ptype
        self.kwargs = kwargs


class FcnArgs(Args):
    """Fully Connected Network arguments"""

    def __init__(self, in_features, out_features, bias=True, activation=None, batch_norm=False, dropout=-1):
        super(FcnArgs, self).__init__(activation, batch_norm, dropout)
        self._args = (in_features, out_features, bias)


class WeaveLayerArgs(Args):

    def __init__(self, n_atom_input_feat=75, n_pair_input_feat=14, n_atom_output_feat=50, n_pair_output_feat=50,
                 n_hidden_AA=50, n_hidden_PA=50, n_hidden_AP=50, n_hidden_PP=50, update_pair=True, activation=None,
                 batch_norm=False, dropout=-1):
        super(WeaveLayerArgs, self).__init__(activation, batch_norm, dropout)
        self._args = (n_atom_input_feat, n_pair_input_feat, n_atom_output_feat, n_pair_output_feat,
                      n_hidden_AA, n_hidden_PA, n_hidden_AP, n_hidden_PP, update_pair, self.activation_function)


class WeaveGatherArgs(Args):
    def __init__(self, conv_out_depth, n_depth=128, gaussian_expand=False, activation='tanh', epsilon=1e-7):
        super(WeaveGatherArgs, self).__init__(activation, False, -1)
        self._args = (conv_out_depth, n_depth, gaussian_expand, self.activation_function, epsilon)


class GraphConvArgs(Args):
    def __init__(self, in_dim, out_dim, min_deg=0, max_deg=10, activation=None, pool=True, batch_norm=True,
                 dropout=-1, dense_layer_size=128):
        super(GraphConvArgs, self).__init__(activation, batch_norm, dropout)
        self.graph_pool = pool
        self.dense_layer_size = dense_layer_size
        self._args = (in_dim, out_dim, min_deg, max_deg, self.activation_function)
