# Author: bbrighttaer
# Project: ivpgan
# Date: 6/17/19
# Time: 1:30 PM
# File: test_dtiviews.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from ivpgan.data.data import load_proteins
from ivpgan.utils.args import PoolingArg, ConvArgs
from ivpgan.utils.calc_dim import calc_1d_outdim
from proj.dti.dtiviews import EcfpView, WeaveView, GraphConvView
from ivpgan.data import load_dti_data, DtiDataset


class DtiViewsTest(unittest.TestCase):

    def setUp(self):
        self.latent_dim = 100
        self.dataset = "davis"
        self.prot_desc_path = ['../../data/davis_data/prot_desc.csv']
        self.prot_info = load_proteins(self.prot_desc_path)

    def test_dim(self):
        conv_args = (ConvArgs(in_channels=1,
                              out_channels=5,
                              kernel_size=5,
                              conv_type='1d',
                              dilation=1,
                              padding=2,
                              stride=1,
                              activation='relu',
                              pooling=PoolingArg(ptype='max_pool',
                                                 padding=2,
                                                 kernel_size=5,
                                                 dilation=1,
                                                 stride=2),
                              batch_norm=False),)
        print(calc_1d_outdim(8677, conv_args))

    def test_ecfp(self):
        ecfp_view = EcfpView(self.latent_dim)
        tasks, all_data, transformers, prot_desc_dict = load_dti_data(featurizer="ECFP", dataset=self.dataset,
                                                                      prot_seq_dict=self.prot_info[1])
        self.assertIsNotNone(all_data)
        print(tasks, len(all_data), [len(d) for d in all_data])
        pair = all_data[0].X[0]
        print(type(pair[0]), type(pair[1]))
        self.assertIsNotNone(ecfp_view.task_model)
        self.assertIsNotNone(ecfp_view.rep_model)
        self.assertIsNotNone(ecfp_view.attn_model)
        model = ecfp_view.model()
        self.assertIsNotNone(model)
        print(type(model), str(model))

    def test_weave(self):
        weave_view = WeaveView(self.latent_dim)
        tasks, all_data, transformers, prot_desc_dict = load_dti_data(featurizer="Weave", dataset=self.dataset,
                                                                      prot_seq_dict=self.prot_info[1])
        self.assertIsNotNone(all_data)
        print(tasks, len(all_data), [len(d) for d in all_data])
        pair = all_data[0].X[0]
        print(type(pair[0]), type(pair[1]))
        self.assertIsNotNone(weave_view.task_model)
        self.assertIsNotNone(weave_view.rep_model)
        self.assertIsNotNone(weave_view.attn_model)
        self.assertIsNotNone(weave_view.weave_model)
        model = weave_view.model()
        self.assertIsNotNone(model)
        print(type(model), str(model))

    def test_gconv(self):
        gconv_view = GraphConvView(self.latent_dim)
        tasks, all_data, transformers, prot_desc_dict = load_dti_data(featurizer="GraphConv", dataset=self.dataset,
                                                                      prot_seq_dict=self.prot_info[1])
        self.assertIsNotNone(all_data)
        print(tasks, len(all_data), [len(d) for d in all_data])
        pair = all_data[0].X[0]
        print(type(pair[0]), type(pair[1]))
        self.assertIsNotNone(gconv_view.task_model)
        self.assertIsNotNone(gconv_view.rep_model)
        self.assertIsNotNone(gconv_view.attn_model)
        self.assertIsNotNone(gconv_view.gconv_model)
        model = gconv_view.model()
        self.assertIsNotNone(model)
        print(type(model), str(model))

    def test_dataset(self):
        tasks, efcp_all_data, transformers, prot_desc_dict = load_dti_data(featurizer="ECFP", dataset=self.dataset,
                                                                           prot_desc_path=self.prot_desc_path)

        tasks, weave_all_data, transformers, prot_desc_dict = load_dti_data(featurizer="Weave", dataset=self.dataset,
                                                                            prot_desc_path=self.prot_desc_path)
        tasks, gconv_all_data, transformers, prot_desc_dict = load_dti_data(featurizer="GraphConv",
                                                                            dataset=self.dataset,
                                                                            prot_desc_path=self.prot_desc_path)
        dataset = DtiDataset(x_s=(efcp_all_data[0].X, weave_all_data[0].X, gconv_all_data[0].X),
                             y_s=(efcp_all_data[0].y, weave_all_data[0].y, gconv_all_data[0].y))
        self.assertEqual(len(dataset), len(efcp_all_data[0]))
        print("Dataset size={}, sample X={}, sample y={}".format(len(dataset), dataset[0][0], dataset[0][1]))
