from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import deepchem
import padme


def load_tc_kinases(featurizer='Weave', cross_validation=False, test=False, split='random',
                    reload=True, K=5, mode='regression', predict_cold=False, cold_drug=False,
                    cold_target=False, split_warm=False, filter_threshold=0, prot_seq_dict=None, oversampled=False):
    # The last parameter means only splitting into training and validation sets.

    if cross_validation:
        assert not test

    if mode == 'regression' or mode == 'reg-threshold':
        mode = 'regression'
        tasks = ['davis', 'metz', 'kiba', 'toxcast_bind']
        file_name = "kinase_tc.csv"
    elif mode == 'classification':
        tasks = ['davis_bin', 'metz_bin', 'kiba_bin', 'toxcast_bind_bin']
        file_name = "kinase_tc_bin.csv"

    data_dir = "synthesized_data/"
    if reload:
        delim = "/"
        if predict_cold:
            delim = "_cold" + delim
        elif split_warm:
            delim = "_warm" + delim
        elif cold_drug:
            delim = "_cold_drug" + delim
        elif cold_target:
            delim = "_cold_target" + delim
        if cross_validation:
            delim = "_CV" + delim
            save_dir = os.path.join(data_dir, featurizer + delim + "kinase_tc/" + mode + "/" + split)
            loaded, all_dataset, transformers = padme.utils.save.load_cv_dataset_from_disk(
                save_dir, K)
        else:
            save_dir = os.path.join(data_dir, featurizer + delim + "kinase_tc/" + mode + "/" + split)
            loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
                save_dir)
        if loaded:
            return tasks, all_dataset, transformers

    dataset_file = os.path.join(data_dir, file_name)
    if featurizer == 'Weave':
        featurizer = padme.feat.WeaveFeaturizer()
    elif featurizer == 'ECFP':
        featurizer = padme.feat.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
        featurizer = padme.feat.ConvMolFeaturizer()

    loader = padme.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", protein_field="proteinName",
        featurizer=featurizer)
    dataset = loader.featurize(dataset_file, shard_size=8192)

    if mode == 'regression':
        transformers = [
            padme.trans.NormalizationTransformer(
                transform_y=True, dataset=dataset)
        ]
    elif mode == 'classification':
        transformers = [
            padme.trans.BalancingTransformer(transform_w=True, dataset=dataset)
        ]

    print("About to transform data")
    for transformer in transformers:
        dataset = transformer.transform(dataset)

    splitters = {
        'index': deepchem.splits.IndexSplitter(),
        'random': padme.splits.RandomSplitter(split_cold=predict_cold, cold_drug=cold_drug,
                                              cold_target=cold_target, split_warm=split_warm,
                                              prot_seq_dict=prot_seq_dict,
                                              threshold=filter_threshold),
        'scaffold': deepchem.splits.ScaffoldSplitter(),
        'butina': deepchem.splits.ButinaSplitter(),
        'task': deepchem.splits.TaskSplitter()
    }
    splitter = splitters[split]
    if test:
        train, valid, test = splitter.train_valid_test_split(dataset)
        all_dataset = (train, valid, test)
        if reload:
            deepchem.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                                     transformers)
    elif cross_validation:
        fold_datasets = splitter.k_fold_split(dataset, K)
        all_dataset = fold_datasets
        if reload:
            padme.utils.save.save_cv_dataset_to_disk(save_dir, all_dataset, K, transformers)

    else:
        # not cross validating, and not testing.
        train, valid, test = splitter.train_valid_test_split(dataset, frac_valid=0.2,
                                                             frac_test=0)
        all_dataset = (train, valid, test)
        if reload:
            deepchem.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                                     transformers)

    return tasks, all_dataset, transformers
