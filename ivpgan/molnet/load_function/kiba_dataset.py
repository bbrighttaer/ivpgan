from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import deepchem
import padme
import pandas as pd

import ivpgan.splits as splits


def load_kiba(featurizer='Weave', cross_validation=False, test=False, split='random',
               reload=True, K=5, mode='regression', predict_cold=False, cold_drug=False,
               cold_target=False, cold_drug_cluster=False, split_warm=False, filter_threshold=0,
               prot_seq_dict=None, currdir="./", oversampled=False, input_protein=True, seed=0):
    # The last parameter means only splitting into training and validation sets.

    if cross_validation:
        assert not test
    data_dir = currdir + "KIBA_data/"
    if input_protein:
        if mode == 'regression' or mode == 'reg-threshold':
            mode = 'regression'
            tasks = ['kiba']
            file_name = "restructured_unique.csv"
            print("Data file:", file_name)
        elif mode == 'classification':
            tasks = ['kiba_bin']
            file_name = "restructured_bin.csv"
    else:
        if mode == 'regression' or mode == 'reg-threshold':
            mode = 'regression'
            file_name = "restructured_no_prot.csv"
        elif mode == 'classification':
            file_name = "restructured_bin_no_prot.csv"
        dataset_file = os.path.join(data_dir, file_name)
        df = pd.read_csv(dataset_file, header=0, index_col=False)
        headers = list(df)
        tasks = headers[:-1]

    if reload:
        delim = "/"
        if not input_protein:
            delim = "_no_prot" + delim
        if filter_threshold > 0:
            delim = "_filtered" + delim
        if predict_cold:
            delim = "_cold" + delim
        elif split_warm:
            delim = "_warm" + delim
        elif cold_drug:
            delim = "_cold_drug" + delim
        elif cold_target:
            delim = "_cold_target" + delim
        elif cold_drug_cluster:
            delim = '_cold_drug_cluster' + delim
        if cross_validation:
            delim = "_CV" + delim
            save_dir = os.path.join(data_dir, featurizer + delim + mode + "/" + split + "_seed_" + str(seed))
            loaded, all_dataset, transformers = padme.utils.save.load_cv_dataset_from_disk(
                save_dir, K)
        else:
            save_dir = os.path.join(data_dir, featurizer + delim + mode + "/" + split + "_seed_" + str(seed))
            loaded, all_dataset, transformers = padme.utils.save.load_dataset_from_disk(
                save_dir)
        if loaded:
            return tasks, all_dataset, transformers

    dataset_file = os.path.join(data_dir, file_name)
    if featurizer == 'Weave':
        featurizer = padme.feat.WeaveFeaturizer()
    elif featurizer == 'ECFP4':
        featurizer = padme.feat.CircularFingerprint(size=1024, radius=2)
    elif featurizer == 'ECFP8':
        featurizer = padme.feat.CircularFingerprint(size=1024, radius=4)
    elif featurizer == 'GraphConv':
        featurizer = padme.feat.ConvMolFeaturizer()

    loader = padme.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", protein_field="proteinName",
        source_field='protein_dataset', featurizer=featurizer, prot_seq_dict=prot_seq_dict)
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
        'random': splits.RandomSplitter(split_cold=predict_cold, cold_drug=cold_drug,
                                        cold_target=cold_target, cold_drug_cluster=cold_drug_cluster,
                                        split_warm=split_warm,
                                        prot_seq_dict=prot_seq_dict, threshold=filter_threshold,
                                        input_protein=input_protein),
        'scaffold': deepchem.splits.ScaffoldSplitter(),
        'butina': deepchem.splits.ButinaSplitter(),
        'task': deepchem.splits.TaskSplitter()
    }
    splitter = splitters[split]
    if test:
        train, valid, test = splitter.train_valid_test_split(dataset, seed=seed)
        all_dataset = (train, valid, test)
        if reload:
            padme.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                                  transformers)
    elif cross_validation:
        fold_datasets = splitter.k_fold_split(dataset, K, seed=seed)
        all_dataset = fold_datasets
        if reload:
            padme.utils.save.save_cv_dataset_to_disk(save_dir, all_dataset, K, transformers)

    else:
        # not cross validating, and not testing.
        train, valid, test = splitter.train_valid_test_split(dataset, frac_train=0.9, frac_valid=0.1,
                                                             frac_test=0, seed=seed)
        all_dataset = (train, valid, test)
        if reload:
            padme.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                                  transformers)

    return tasks, all_dataset, transformers
