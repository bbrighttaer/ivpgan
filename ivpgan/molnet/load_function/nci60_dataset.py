from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import deepchem
import padme
import pandas as pd


# This dataset is for prediction only, there is no true values known.
def load_nci60(featurizer='Weave', cross_validation=False, test=False,
               split='random', reload=True, K=5, mode='regression', predict_cold=False,
               cold_drug=False, cold_target=False, split_warm=False, filter_threshold=0,
               prot_seq_dict=None, oversampled=False):
    # data_to_train = 'tc'
    data_to_train = 'davis'

    if mode == 'regression' or mode == 'reg-threshold':
        mode = 'regression'
        file_name = "AR_ER_intxn_s"
        # substitute file_name with the template prediction file you have. For example, if your
        # data_to_train is 'davis', you should have a "_davis" as the suffix of your template
        # prediction file name, like "restructured_template_davis.csv"; in this case the file_name
        # variable should be "restructured_template".

    elif mode == 'classification':
        file_name = "restructured_bin"

    file_name = file_name + "_" + data_to_train + '.csv'
    # data_dir = "NCI60_data/"
    data_dir = "davis_data/"
    # HACK: The last line is only temporary.
    dataset_file = os.path.join(data_dir, file_name)
    df = pd.read_csv(dataset_file, header=0, index_col=False)
    headers = list(df)
    # tasks = headers[:-3]
    # I commented the last line out to make it less prone to errors, should the header orders change.
    headers.remove("proteinName")
    headers.remove("protein_dataset")
    headers.remove("smiles")
    tasks = headers

    if reload:
        delim = "_" + data_to_train + "/"
        save_dir = os.path.join(data_dir, featurizer + delim + mode + "/" + split)
        loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
            save_dir)
        if loaded:
            return tasks, all_dataset, transformers

    # HACK: the following if-else block could be prone to errors.
    if data_to_train == "tc":
        loaded, _, transformers = padme.molnet.load_toxcast(featurizer=featurizer, split=split,
                                                            cross_validation=False, reload=True, mode=mode)
    elif data_to_train == "davis":
        loaded, _, transformers = padme.molnet.load_davis(featurizer=featurizer, split=split,
                                                          cross_validation=True, reload=True, mode=mode,
                                                          filter_threshold=0)
        # NOTE: tweak the parameters such that it suits your use case.
    elif data_to_train == "kiba":
        loaded, _, transformers = padme.molnet.load_kiba(featurizer=featurizer, split=split,
                                                         cross_validation=False, reload=True, mode=mode,
                                                         split_warm=True, filter_threshold=6)
    else:
        assert False

    assert loaded

    if featurizer == 'Weave':
        featurizer = padme.feat.WeaveFeaturizer()
    elif featurizer == 'ECFP':
        featurizer = padme.feat.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
        featurizer = padme.feat.ConvMolFeaturizer()

    loader = padme.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", protein_field="proteinName",
        source_field='protein_dataset', featurizer=featurizer, prot_seq_dict=prot_seq_dict)
    dataset = loader.featurize(dataset_file, shard_size=8192)

    # print("About to transform data")
    # for transformer in transformers:
    #   dataset = transformer.transform(dataset)

    splitters = {
        'index': deepchem.splits.IndexSplitter(),
        'random': padme.splits.RandomSplitter(),
        'scaffold': deepchem.splits.ScaffoldSplitter(),
        'butina': deepchem.splits.ButinaSplitter(),
        'task': deepchem.splits.TaskSplitter()
    }
    splitter = splitters[split]

    # HACK: We set frac_train to 1.0 because assume NCI60 dataset is for prediction only: there
    # is no underlying truth. To predict all drug-target pairs, we need to let all samples be in
    # the "training" set, though it is a misnomer.
    train, valid, test = splitter.train_valid_test_split(dataset, frac_train=1.0,
                                                         frac_valid=0.0, frac_test=0)
    all_dataset = (train, valid, test)
    if reload:
        deepchem.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                                 transformers)

    return tasks, all_dataset, transformers
