# Author: bbrighttaer
# Project: ivpgan
# Date: 5/24/19
# Time: 2:50 PM
# File: test_layers.py


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import deepchem as dc
import padme
import pandas as pd
import rdkit.Chem as ch
import torch
from torch.autograd import Variable, gradcheck

from ivpgan.nn.layers import WeaveLayer, WeaveGather, GraphConvLayer, GraphPool, GraphGather
from ivpgan.utils.mols import process_weave_input, process_graph_conv_input


class TestLayers(unittest.TestCase):
    def setUp(self):
        # import os
        #
        # cwd = os.getcwd()  # Get the current working directory (cwd)
        # print(cwd)
        # files = os.listdir(cwd)  # Get all the files in that directory
        # print("Files in '%s': %s" % (cwd, files))

        self.df = pd.read_csv('./data/davis_50.csv')[:3]
        self.smiles = self.df['smiles']
        self.mols = [ch.MolFromSmiles(s) for s in self.smiles]

    def test_weave_layer(self):
        feat_weave = padme.feat.WeaveFeaturizer()
        mols_feat = feat_weave(self.mols)
        self.assertEqual(len(mols_feat), len(self.mols))
        atom_features, pair_features, pair_split, atom_split, atom_to_pair = process_weave_input(mols_feat)
        print("\natom_features={}, pair_features={}, pair_split={}, atom_split={}, atom_to_pair={}\n".format(
            atom_features.shape, pair_features.shape, pair_split.shape, atom_split.shape, atom_to_pair.shape
        ))
        weave1 = WeaveLayer(n_atom_input_feat=75, n_pair_input_feat=14, n_atom_output_feat=50, n_pair_output_feat=50,
                            n_hidden_AA=50, n_hidden_PA=50, n_hidden_AP=50, n_hidden_PP=50, update_pair=True,
                            activation='relu')

        weave2 = WeaveLayer(n_atom_input_feat=50, n_pair_input_feat=50, n_atom_output_feat=50, n_pair_output_feat=50,
                            n_hidden_AA=50, n_hidden_PA=50, n_hidden_AP=50, n_hidden_PP=50, update_pair=True,
                            activation='relu')
        # conv 1
        input_data = [atom_features, pair_features, pair_split, atom_split, atom_to_pair]
        w1_outputs = weave1(input_data)
        self.assertIsNotNone(w1_outputs, msg='weave 1 conv output is none')
        print('A.shape={}, P.shape={}'.format(w1_outputs[0].shape, w1_outputs[1].shape))

        # conv 2
        input_data = [w1_outputs[0], w1_outputs[1], pair_split, atom_split, atom_to_pair]
        w2_outputs = weave2(input_data)
        self.assertIsNotNone(w2_outputs, msg='weave 2 conv output is none')
        print('A.shape={}, P.shape={}'.format(w2_outputs[0].shape, w2_outputs[1].shape))

        A = torch.nn.Dropout(.25)(w1_outputs[0])
        A = torch.nn.BatchNorm1d(50)(A)

        weave_gather = WeaveGather(conv_out_depth=50, gaussian_expand=True)
        output = weave_gather([A, atom_split])
        self.assertIsNotNone(output, msg="Output is none")
        print("Output shape=", output.shape)

        params = list(weave1.parameters())
        print('\n# of trainable parameters =', len(params))
        for p in params:
            print(p.size())

    def test_graph_conv_layer(self):
        feat_graph_conv = padme.feat.ConvMolFeaturizer()
        mols = feat_graph_conv(self.mols)
        self.assertEqual(len(mols), len(self.mols))
        input_data = process_graph_conv_input(mols)
        num_atom_feat = input_data[0][-1].shape[0]
        gconv1 = GraphConvLayer(in_dim=num_atom_feat, out_dim=64)

        # conv 1
        out1 = gconv1(input_data)
        self.assertIsNotNone(out1, msg="graph convolution 1 returned None")
        out1 = torch.nn.BatchNorm1d(64)(out1)
        print('\ngraph conv shape=', out1.shape)

        # conv pool
        gconvp1 = GraphPool()
        input_data[0] = out1
        out1 = gconvp1(input_data)
        self.assertIsNotNone(out1, msg="graph pool 1 returned None")
        print('conv pool shape=', out1.shape)

        # conv gather
        gconv_gather = GraphGather()
        input_data[0] = out1
        out_gathered = gconv_gather(input_data, len(mols))
        self.assertIsNotNone(out_gathered, "Nothing was gathered")
        print('out_gathered shape=', out_gathered.shape)

        params = list(gconv1.parameters())
        print('\n# of trainable parameters =', len(params))
        for p in params:
            print(p.size())

        # backprop test
        idx = 1  # np.random.randint(10)
        self.assertIsNone(params[idx].grad, "Gradient property is not none")
        c = 2
        o = c * torch.sum(out_gathered)
        o.backward()
        self.assertIsNotNone(params[idx].grad, "Backprop failed")

        for i in range(len(params)):
            print('parameter-{} (shape={}) gradient = {}'.format(i + 1, params[i].shape, params[i].grad))