# Author: bbrighttaer
# Project: ivpgan
# Date: 5/20/19
# Time: 1:17 PM
# File: data.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re

import numpy as np
import pandas as pd
import torch
from padme.feat.mol_graphs import ConvMol
from torch.utils.data import dataset as ds

from ivpgan.molnet.load_function.davis_dataset import load_davis
from ivpgan.molnet.load_function.kiba_dataset import load_kiba
from ivpgan.molnet.load_function.kinase_datasets import load_kinases
from ivpgan.molnet.load_function.metz_dataset import load_metz
from ivpgan.molnet.load_function.nci60_dataset import load_nci60
from ivpgan.molnet.load_function.tc_dataset import load_toxcast
from ivpgan.molnet.load_function.tc_full_kinase_datasets import load_tc_full_kinases
from ivpgan.molnet.load_function.tc_kinase_datasets import load_tc_kinases


def load_prot_dict(prot_desc_dict, prot_seq_dict, prot_desc_path,
                   sequence_field, phospho_field):
    if re.search('davis', prot_desc_path, re.I):
        source = 'davis'
    elif re.search('metz', prot_desc_path, re.I):
        source = 'metz'
    elif re.search('kiba', prot_desc_path, re.I):
        source = 'kiba'
    elif re.search('toxcast', prot_desc_path, re.I):
        source = 'toxcast'

    df = pd.read_csv(prot_desc_path, index_col=0)
    # protList = list(df.index)
    for row in df.itertuples():
        descriptor = row[2:]
        descriptor = np.array(descriptor)
        descriptor = np.reshape(descriptor, (1, len(descriptor)))
        pair = (source, row[0])
        assert pair not in prot_desc_dict
        prot_desc_dict[pair] = descriptor
        sequence = row[sequence_field]
        phosphorylated = row[phospho_field]
        assert pair not in prot_seq_dict
        prot_seq_dict[pair] = (phosphorylated, sequence)


def load_dti_data(featurizer, dataset, prot_seq_dict, input_protein=True, cross_validation=False, test=False,
                  fold_num=5, split='random', reload=True, predict_cold=False, cold_drug=False, cold_target=False,
                  cold_drug_cluster=False, split_warm=False, filter_threshold=0,
                  mode='regression', data_dir='../../data/', seed=0):
    loading_functions = {
        'davis': load_davis,
        'metz': load_metz,
        'kiba': load_kiba,
        'toxcast': load_toxcast,
        'all_kinase': load_kinases,
        'tc_kinase': load_tc_kinases,
        'tc_full_kinase': load_tc_full_kinases,
        'nci60': load_nci60
    }

    if cross_validation:
        test = False
    tasks, all_dataset, transformers = loading_functions[dataset](featurizer=featurizer,
                                                                  cross_validation=cross_validation,
                                                                  test=test, split=split, reload=reload,
                                                                  K=fold_num, mode=mode,
                                                                  predict_cold=predict_cold,
                                                                  cold_drug=cold_drug,
                                                                  cold_target=cold_target,
                                                                  cold_drug_cluster=cold_drug_cluster,
                                                                  split_warm=split_warm,
                                                                  prot_seq_dict=prot_seq_dict,
                                                                  filter_threshold=filter_threshold,
                                                                  input_protein=input_protein,
                                                                  currdir=data_dir,
                                                                  seed=seed, )
    return tasks, all_dataset, transformers


def load_proteins(prot_desc_path):
    """
    Retrieves all proteins in the tuple of paths given.

    :param prot_desc_path: A tuple of file paths containing the protein (PSC) descriptors.
    :return: A set of dicts: (descriptor information, sequence information)
    """
    prot_desc_dict = {}
    prot_seq_dict = {}
    for path in prot_desc_path:
        load_prot_dict(prot_desc_dict, prot_seq_dict, path, 1, 2)
    return prot_desc_dict, prot_seq_dict


class DtiDataset(ds.Dataset):

    def __init__(self, x_s, y_s, w_s):
        """
        Creates a Drug-Target Indication dataset object.

        :param x_s: a tuple of X data of each view.
        :param y_s: a tuple of y data of each view.
        :param w_s: a tuple of label weights of each view.
        """
        assert len(x_s) == len(y_s) == len(w_s), "Number of views in x_s must be equal to that of y_s."
        self.x_s = x_s
        self.y_s = y_s
        self.w_s = w_s

    def __len__(self):
        return len(self.x_s[0])

    def __getitem__(self, index):
        x_s = []
        y_s = []
        w_s = []
        for view_x, view_y, view_w in zip(self.x_s, self.y_s, self.w_s):
            x_s.append(view_x[index])
            y_s.append(view_y[index])
            w_s.append(view_w[index])
        return x_s, y_s, w_s


class Dataset(ds.Dataset):
    """Wrapper for the dataset to pytorch models"""

    def __init__(self, views_data):
        """
        Creates a dataset wrapper.

        :param views_data: Data of all views. Structure: ((X1, Y1), (X2, Y2), ...)
        """
        self.X_list = []
        self.y_list = []
        self.num_views = len(views_data)
        for data in views_data:
            self.X_list.append(data[0])  # 0 -> X data
            self.y_list.append(data[1])  # 1 -> y data
        super(Dataset, self).__init__()

    def __len__(self):
        return len(self.X_list[0])

    def __getitem__(self, index):
        ret_ds = []
        for i in range(self.num_views):
            x = self.X_list[i][index]
            y = self.y_list[i][index]
            ret_ds.append((torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.long)))
        return ret_ds


def batch_collator(batch, prot_desc_dict, spec):
    batch = np.array(batch)  # batch.shape structure: (batch_size, x-0/y-1/w-2 data, view index)
    data = {}
    # num_active_views = reduce(lambda x1, x2: x1 + x2, flags.values())
    funcs = {
        "ecfp4": process_ecfp_view_data,
        "ecfp8": process_ecfp_view_data,
        "weave": process_weave_view_data,
        "gconv": process_gconv_view_data
    }
    active_views = []
    if isinstance(spec, dict):
        for k in spec:
            if spec[k]:
                active_views.append(k)
    else:
        active_views.append(spec)
    for i, v_name in enumerate(active_views):
        func = funcs[v_name]
        data[v_name] = (func(batch, prot_desc_dict, i), batch[:, 1, i], batch[:, 2, i])
    return len(batch), data


def process_ecfp_view_data(X, prot_desc_dict, idx):
    """
    Converts ECFP-Protein pair dataset to a pytorch tensor.

    :param X:
    :param prot_desc_dict:
    :return:
    """
    mols_tensor = prots_tensor = None
    if X is not None:
        x_data = X[:, 0, idx]
        mols = [pair[0] for pair in x_data]
        mols_tensor = torch.from_numpy(np.array([mol.get_array() for mol in mols]))
        prots = [pair[1] for pair in x_data]
        prot_names = [prot.get_name() for prot in prots]
        prot_desc = [prot_desc_dict[prot_name] for prot_name in prot_names]
        prot_desc = np.array(prot_desc)
        prot_desc = prot_desc.reshape(prot_desc.shape[0], prot_desc.shape[2])
        prots_tensor = torch.from_numpy(prot_desc)
    return cuda(mols_tensor.float()), cuda(prots_tensor.float())


def process_weave_view_data(X, prot_desc_dict, idx):
    """
    Converts Weave-Protein pair dataset to a pytorch tensor.

    :param X:
    :param prot_desc_dict:
    :return:
    """
    atom_feat = []
    pair_feat = []
    atom_split = []
    atom_to_pair = []
    pair_split = []
    prot_descriptor = []
    start = 0
    x_data = X[:, 0, idx]
    for im, pair in enumerate(x_data):
        mol, prot = pair
        n_atoms = mol.get_num_atoms()
        # number of atoms in each molecule
        atom_split.extend([im] * n_atoms)
        # index of pair features
        C0, C1 = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
        atom_to_pair.append(
            np.transpose(
                np.array([C1.flatten() + start,
                          C0.flatten() + start])))
        # number of pairs for each atom
        pair_split.extend(C1.flatten() + start)
        start = start + n_atoms

        # atom features
        atom_feat.append(mol.get_atom_features())
        # pair features
        n_pair_feat = mol.pairs.shape[2]
        pair_feat.append(
            np.reshape(mol.get_pair_features(),
                       (n_atoms * n_atoms, n_pair_feat)))
        prot_descriptor.append(prot_desc_dict[prot.get_name()])
    prots_tensor = torch.from_numpy(np.concatenate(prot_descriptor, axis=0))
    mol_data = [
        cuda(torch.tensor(np.concatenate(atom_feat, axis=0), dtype=torch.float)),
        cuda(torch.tensor(np.concatenate(pair_feat, axis=0), dtype=torch.float)),
        cuda(torch.tensor(np.array(pair_split), dtype=torch.int)),
        cuda(torch.tensor(np.array(atom_split), dtype=torch.int)),
        cuda(torch.tensor(np.concatenate(atom_to_pair, axis=0), dtype=torch.long))
    ]
    return mol_data, cuda(prots_tensor.float())


def process_gconv_view_data(X, prot_desc_dict, idx):
    """
    Converts Graph convolution-Protein pair dataset to a pytorch tensor.

    :param X:
    :param prot_desc_dict:
    :return:
    """
    mol_data = []
    x_data = X[:, 0, idx]
    mols = [pair[0] for pair in x_data]
    multiConvMol = ConvMol.agglomerate_mols(mols)
    mol_data.append(cuda(torch.from_numpy(multiConvMol.get_atom_features())))
    mol_data.append(cuda(torch.from_numpy(multiConvMol.deg_slice)))
    mol_data.append(cuda(torch.tensor(multiConvMol.membership)))
    for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
        mol_data.append(cuda(torch.from_numpy(multiConvMol.get_deg_adjacency_lists()[i])))

    # protein
    prots = [pair[1] for pair in x_data]
    prot_names = [prot.get_name() for prot in prots]
    prot_desc = [prot_desc_dict[prot_name] for prot_name in prot_names]
    prot_desc = np.array(prot_desc)
    prot_desc = prot_desc.reshape(prot_desc.shape[0], prot_desc.shape[2])
    prots_tensor = cuda(torch.from_numpy(prot_desc))

    batch_size = len(x_data)
    return mol_data, prots_tensor.float()


def cuda(tensor):
    from ivpgan import cuda
    if cuda:
        return tensor.cuda()
    else:
        return tensor


def get_data(featurizer, flags, prot_sequences, seed):
    # logger = get_logger(name="Data loader")
    print("--------------About to load {}-{} data-------------".format(featurizer, flags['dataset']))
    try:
        return load_dti_data(featurizer=featurizer,
                             dataset=flags['dataset'],
                             prot_seq_dict=prot_sequences,
                             input_protein=True,
                             cross_validation=flags['cv'],
                             test=flags['test'],
                             fold_num=flags['fold_num'],
                             split=flags['splitting_alg'],
                             reload=flags['reload'],
                             predict_cold=flags['predict_cold'],
                             cold_drug=flags['cold_drug'],
                             cold_target=flags['cold_target'],
                             mode='regression',
                             data_dir=flags['data_dir'],
                             cold_drug_cluster=flags['cold_drug_cluster'],
                             split_warm=flags['split_warm'],
                             seed=seed,
                             filter_threshold=flags["filter_threshold"], )
    finally:
        print("--------------{}-{} data loaded-------------".format(featurizer, flags['dataset']))
