# Author: bbrighttaer
# Project: ivpgan
# Date: 5/29/19
# Time: 4:19 PM
# File: models.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

import torch
import torch.nn as nn
import torch.nn.init as init

from ivpgan.nn.layers import Linear, Conv1d, Conv2d
from ivpgan.nn.layers import WeaveGather, WeaveLayer, GraphConvLayer, GraphGather, GraphPool, WeaveBatchNorm, \
    WeaveDropout

relu_batch_norm = False


def get_weights_init(a=5):
    def init_func(m):
        """
        Initializes the trainable parameters.

        :param m: The submodule object
        """
        if isinstance(m, Linear) or isinstance(m, Conv1d) or isinstance(m, Conv2d):
            # if m.activation_name:
            #     func_name = m.activation_name.split('(')[0].lower()
            #     if func_name in ['sigmoid', 'tanh']:
            #         init.xavier_uniform_(m.weight)
            #     else:
            #         init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            # else:
            init.kaiming_uniform_(m.weight, a=math.sqrt(a))
            if m.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, -bound, bound)
                # init.constant_(m.bias, 0)

    return init_func


def create_conv_layers(conv_args):
    layers = []
    for conv_arg in conv_args:
        if conv_arg.conv_type in ["1D", '1d']:
            conv = Conv1d(*conv_arg.args)
            layers.append(conv)

            # Batch normalization
            if conv_arg.use_batch_norm:
                bn = nn.BatchNorm1d(conv_arg[1])
                layers.append(bn)

            # Activation
            if conv_arg.activation_function:
                conv.activation_name = str(conv_arg.activation_function)
                if relu_batch_norm:
                    # if batch norm + relu, do batch norm after applying relu.
                    if conv_arg.use_batch_norm and 'relu' in conv.activation_name.lower():
                        # bn = layers.pop()
                        layers.append(conv_arg.activation_function)
                        # layers.append(bn)
                    else:
                        layers.append(conv_arg.activation_function)
                else:
                    layers.append(conv_arg.activation_function)

            # Dropout
            if conv_arg.dropout > 0:
                dr = nn.Dropout(conv_arg.dropout)
                layers.append(dr)

            # pooling
            if conv_arg.pooling:
                pool = {'max_pool': lambda kwargs: nn.MaxPool1d(**kwargs),
                        'avg_pool': lambda kwargs: nn.AvgPool1d(**kwargs)
                        }.get(conv_arg.pooling.ptype.lower(), lambda x: None)(conv_arg.pooling.kwargs)
                if pool:
                    layers.append(pool)

        elif conv_arg.conv_type in ["2D", '2d']:
            conv = Conv2d(*conv_arg.args)
            layers.append(conv)

            # Batch normalization
            if conv_arg.use_batch_norm:
                bn = nn.BatchNorm2d(conv_arg[1])
                layers.append(bn)

            # Activation
            if conv_arg.activation_function:
                conv.activation_name = str(conv_arg.activation_function)
                if relu_batch_norm:
                    # if batch norm + relu, do batch norm after applying relu.
                    if conv_arg.use_batch_norm and 'relu' in conv.activation_name.lower():
                        # bn = layers.pop()
                        layers.append(conv_arg.activation_function)
                        # layers.append(bn)
                    else:
                        layers.append(conv_arg.activation_function)
                else:
                    layers.append(conv_arg.activation_function)

            # Dropout
            if conv_arg.dropout > 0:
                dr = nn.Dropout2d(conv_arg.dropout)
                layers.append(dr)

            # pooling
            if conv_arg.pooling:
                pool = {'max_pool': lambda kwargs: nn.MaxPool2d(**kwargs),
                        'avg_pool': lambda kwargs: nn.AvgPool2d(**kwargs)
                        }.get(conv_arg.pooling.ptype.lower(), None)(conv_arg.pooling.kwargs)
                if pool:
                    layers.append(pool)
    return layers


def create_fcn_layers(fcn_args):
    layers = []
    for fcn_arg in fcn_args:
        assert fcn_arg.args[
                   1] > 0, "Output layer nodes number must be specified for hidden layers."
        linear = Linear(*fcn_arg.args)
        layers.append(linear)

        # Batch normalization
        if fcn_arg.use_batch_norm:
            bn = nn.BatchNorm1d(fcn_arg[1])
            layers.append(bn)

        # Activation
        if fcn_arg.activation_function:
            linear.activation_name = str(fcn_arg.activation_function)
            if relu_batch_norm:
                # if batch norm + relu, do batch norm after applying relu.
                if fcn_arg.use_batch_norm and 'relu' in linear.activation_name.lower():
                    # bn = layers.pop()
                    layers.append(fcn_arg.activation_function)
                    # layers.append(bn)
                else:
                    layers.append(fcn_arg.activation_function)
            else:
                layers.append(fcn_arg.activation_function)

        # Dropout
        if fcn_arg.dropout > 0:
            dr = nn.Dropout(fcn_arg.dropout)
            layers.append(dr)
    return layers


def create_weave_layers(weave_args):
    layers = []
    for weave_arg in weave_args:
        weave = WeaveLayer(*weave_arg.args)
        layers.append(weave)

        # Batch normalization
        if weave_arg.use_batch_norm:
            bn = WeaveBatchNorm(atom_dim=weave_arg[2], pair_dim=weave_arg[3])
            layers.append(bn)

        # Dropout
        if weave_arg.dropout > 0:
            dr = WeaveDropout(weave_arg.dropout)
            layers.append(dr)
    return layers


def create_graph_conv_layers(gconv_args):
    layers = []
    for gc_arg in gconv_args:
        gconv = GraphConvLayer(*gc_arg.args)
        layers.append(gconv)

        # Batch normalization
        if gc_arg.use_batch_norm:
            bn = nn.BatchNorm1d(gc_arg[1])
            layers.append(bn)

        # Dropout
        if gc_arg.dropout > -1:
            dr = nn.Dropout(gc_arg.dropout)
            layers.append(dr)

        # Pooling
        if gc_arg.graph_pool:
            p = GraphPool(gc_arg[2], gc_arg[3])
            layers.append(p)

        # Dense layer & normalization & dropout
        layers.append(nn.Linear(gc_arg[1], gc_arg.dense_layer_size))
        layers.append(nn.BatchNorm1d(gc_arg.dense_layer_size))
        if gc_arg.dropout > -1:
            layers.append(nn.Dropout(gc_arg.dropout))

        # Gather
        layers.append(GraphGather())
    return layers


class WeaveModel(nn.Module):

    def __init__(self, weave_args, weave_gath_arg):
        """
        Creates a weave model

        :param weave_args: A list of weave arguments.
        :param weave_gath_arg: A weave gather argument.
        """
        super(WeaveModel, self).__init__()
        layers = create_weave_layers(weave_args)
        weave_gath = WeaveGather(*weave_gath_arg.args)
        layers.append(weave_gath)
        self.weave = WeaveSequential(*layers)
        # in_dim = weave_args[-1][2]
        # out_dim = weave_gath_arg[1]
        # self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, input):
        """

        :param input: The input structure is: [atom_features, pair_features, pair_split, atom_split, atom_to_pair]
        :return: Features of molecules.
        """
        output = self.weave(input)
        return output


class WeaveSequential(nn.Sequential):

    def __init__(self, *args):
        super(WeaveSequential, self).__init__(*args)

    def forward(self, input):
        """
        Forward propagation through all attached layers.

        :param input: The input structure is: [atom_features, pair_features, pair_split, atom_split, atom_to_pair]
        :return: A tuple of atom features and pair features of the last weave layer. (A, P)
        """
        input = list(input)
        A = P = None
        for module in self._modules.values():
            if A is not None:
                input[0] = A
            if P is not None:
                input[1] = P
            if isinstance(module, WeaveBatchNorm) or isinstance(module, WeaveDropout):
                A, P = module(A, P)
            elif isinstance(module, WeaveGather):
                return module([A, input[3]])  # returns the molecule features
            else:
                A, P = module(input)
        return A, P


class GraphConvModel(nn.Module):

    def __init__(self, conv_args):
        """
        Creates a graph convolution model.

        :param conv_args: a list of convolution layer arguments.
        """
        super(GraphConvModel, self).__init__()
        self.model = GraphConvSequential(*create_graph_conv_layers(conv_args))

    def forward(self, *input):
        """

        :param input: The structure: [standard graph conv list, batch size]
        :return: molecule(s) features.
        """
        output = self.model(*input)
        return output


class GraphConvSequential(nn.Sequential):

    def __init__(self, *args):
        super(GraphConvSequential, self).__init__(*args)

    def forward(self, input):
        batch_size = input[1]
        input = input[0]
        input = list(input)
        for module in self._modules.values():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.Dropout) \
                    or isinstance(module, nn.Linear) or isinstance(module, nn.ReLU):
                input[0] = module(input[0])
            elif isinstance(module, GraphGather):
                input[0] = module(input, batch_size)
            else:
                input[0] = module(input)
        return input[0]


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


class PairSequential(nn.Module):
    """Handy approach to manage protein and compound models"""

    def __init__(self, mod1: tuple, mod2: tuple, civ_dim=1):
        super(PairSequential, self).__init__()
        self.comp_tup = nn.ModuleList(mod1)
        self.prot_tup = nn.ModuleList(mod2)
        self.civ = CIV(dim=civ_dim)

    def forward(self, inputs):
        comp_input, prot_input = inputs

        # compound procession
        comp_out = comp_input
        for module in self.comp_tup:
            comp_out = module(comp_out)

        # protein processing
        prot_out = prot_input
        for module in self.prot_tup:
            prot_out = module(prot_out)

        # form a single representation
        output = self.civ((comp_out, prot_out))
        return output


class NonsatActivation(nn.Module):
    def __init__(self, ep=1e-4, max_iter=100):
        super(NonsatActivation, self).__init__()
        self.ep = ep
        self.max_iter = max_iter

    def forward(self, x):
        return nonsat_activation(x, self.ep, self.max_iter)


def nonsat_activation(x, ep=1e-4, max_iter=100):
    """
    Implementation of the Non-saturating nonlinearity described in http://proceedings.mlr.press/v28/andrew13.html

    :param x: float, tensor
        Function input
    :param ep:float, optional
        Stop condition reference point.
    :param max_iter: int, optional,
        Helps to avoid infinite iterations.
    :return:
    """
    y = x.detach().clone()
    i = 0
    while True:
        y_ = (2. * y ** 3. / 3. + x) / (y ** 2. + 1.)
        if torch.mean(torch.abs(y_ - y)) <= ep or i > max_iter:
            return y_
        else:
            i += 1
            y = y_.detach()
