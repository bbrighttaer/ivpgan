# Author: bbrighttaer
# Project: ivpgan
# Date: 6/3/19
# Time: 11:51 PM
# File: mols.py


from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import torch
from padme.feat.mol_graphs import ConvMol


def process_weave_input(mols):
    atom_feat = []
    pair_feat = []
    atom_split = []
    atom_to_pair = []
    pair_split = []
    start = 0
    for im, mol in enumerate(mols):
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
    inputs = [
        torch.tensor(np.concatenate(atom_feat, axis=0), dtype=torch.float),
        torch.tensor(np.concatenate(pair_feat, axis=0), dtype=torch.float),
        torch.tensor(np.array(pair_split), dtype=torch.int),
        torch.tensor(np.array(atom_split), dtype=torch.int),
        torch.tensor(np.concatenate(atom_to_pair, axis=0), dtype=torch.long)
    ]
    return inputs


def process_graph_conv_input(mols):
    d = []
    multiConvMol = ConvMol.agglomerate_mols(mols)
    d.append(torch.from_numpy(multiConvMol.get_atom_features()))
    d.append(torch.from_numpy(multiConvMol.deg_slice))
    d.append(torch.tensor(multiConvMol.membership))
    for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
        d.append(torch.from_numpy(multiConvMol.get_deg_adjacency_lists()[i]))
    return d
