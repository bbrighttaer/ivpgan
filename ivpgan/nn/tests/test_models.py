# Author: bbrighttaer
# Project: ivpgan
# Date: 6/1/19
# Time: 11:40 PM
# File: test_models.py


from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import unittest
import torch
import torch.nn as nn
from ivpgan.nn.layers import Linear
from ivpgan.nn.models import WeaveModel, GraphConvModel
from ivpgan.utils.args import FcnArgs, ConvArgs, PoolingArg, WeaveGatherArgs, WeaveLayerArgs, GraphConvArgs
import pandas as pd
import deepchem as dc
import rdkit.Chem as ch

from ivpgan.utils.mols import process_weave_input, process_graph_conv_input


class TestModels(unittest.TestCase):

    def setUp(self):
        self.batch_size = 3
        self.num_features = 256
        self.channels = 1
        self.data_1d = torch.randn(self.batch_size, self.channels, self.num_features)
        self.data_2d = torch.randn(self.batch_size, self.channels, self.num_features, 3)

        self.df = pd.read_csv('./data/davis_50.csv')[:3]
        self.smiles = self.df['smiles']
        self.mols = [ch.MolFromSmiles(s) for s in self.smiles]

    def test_weave_model(self):
        feat_weave = dc.feat.WeaveFeaturizer()
        mols_feat = feat_weave(self.mols)
        self.assertEqual(len(mols_feat), len(self.mols))
        atom_features, pair_features, pair_split, atom_split, atom_to_pair = process_weave_input(mols_feat)
        print("\natom_features={}, pair_features={}, pair_split={}, atom_split={}, atom_to_pair={}\n".format(
            atom_features.shape, pair_features.shape, pair_split.shape, atom_split.shape, atom_to_pair.shape
        ))
        input = [atom_features, pair_features, pair_split, atom_split, atom_to_pair]
        weave_args = (WeaveLayerArgs(n_atom_input_feat=75,
                                     n_pair_input_feat=14,
                                     n_atom_output_feat=50,
                                     n_pair_output_feat=50,
                                     n_hidden_AA=50,
                                     n_hidden_PA=50,
                                     n_hidden_AP=50,
                                     n_hidden_PP=50,
                                     update_pair=True,
                                     activation='relu'),
                      WeaveLayerArgs(n_atom_input_feat=50,
                                     n_pair_input_feat=50,
                                     n_atom_output_feat=50,
                                     n_pair_output_feat=50,
                                     n_hidden_AA=50,
                                     n_hidden_PA=50,
                                     n_hidden_AP=50,
                                     n_hidden_PP=50,
                                     update_pair=True,
                                     batch_norm=True,
                                     dropout=.2,
                                     activation='relu'))
        wg_args = WeaveGatherArgs(conv_out_depth=50, gaussian_expand=True, n_depth=128)

        model = WeaveModel(weave_args, wg_args)
        output = model(input)
        self.assertIsNotNone(output, "Weave output is none")
        print("Weave output shape=", output.shape)
        for param in model.parameters():
            print(param.shape)

    def test_graph_conv_model(self):
        feat_graph_conv = dc.feat.ConvMolFeaturizer()
        mols = feat_graph_conv(self.mols)
        self.assertEqual(len(mols), len(self.mols))
        input_data = process_graph_conv_input(mols)
        num_atom_feat = input_data[0][-1].shape[0]
        conv_args = (GraphConvArgs(in_dim=num_atom_feat,
                                   out_dim=128,
                                   dropout=.2),)
        model = GraphConvModel(conv_args)
        output = model(input_data, len(mols))
        self.assertIsNotNone(output, "Graph convolution output is none")
        print("Graph convolution output shape=", output.shape)
        for param in model.parameters():
            print(param.shape)
