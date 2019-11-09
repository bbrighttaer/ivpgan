# Author: bbrighttaer
# Project: ivpgan
# Date: 5/20/19
# Time: 2:47 PM
# File: train_helpers.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gzip
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from keras.utils.data_utils import get_file
from sklearn import svm as svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from ivpgan import cuda


def split_mnist(dataset, targets, num_views=2):
    shape = dataset.shape
    view_dim = shape[1] // num_views
    views_data = []

    last_index = 0
    for v in range(num_views):
        v_data = dataset[:, last_index:last_index + view_dim]
        views_data.append((v_data, targets))
        last_index = last_index + view_dim

    return views_data


def train_svm(train_x, train_y):
    print('training SVM...')
    clf = svm.LinearSVC(C=0.01, dual=False)
    clf.fit(train_x, train_y)

    # sanity check
    p = clf.predict(train_x)
    san_acc = accuracy_score(train_y, p)

    return san_acc, clf


def svm_classify(data, C=0.01):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    train_data, train_label = data[0]
    test_data, test_label = data[1]

    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label)

    # sanity check
    p = clf.predict(train_data)
    san_acc = accuracy_score(train_label, p)

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)
    return san_acc, test_acc


def construct_iris_views(dataframe):
    view1 = dataframe[['SepalLengthCm', 'SepalWidthCm', 'label']].values
    view2 = dataframe[['PetalLengthCm', 'PetalWidthCm', 'label']].values
    return view1, view2


def load_data(data_file, url, normalize=True):
    """loads the data from the gzip pickled files, and converts to numpy arrays"""
    print('loading data ...')
    path = get_file(data_file, origin=url)
    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = load_pickle(f)
    f.close()

    train_set_x, train_set_y = make_numpy_array(train_set)
    # train_set_x = train_set_x / 255.
    valid_set_x, valid_set_y = make_numpy_array(valid_set)
    # valid_set_x = valid_set_x / 255.
    train_set_x = np.vstack([train_set_x, valid_set_x])
    train_set_y = np.vstack([train_set_y.reshape(-1, 1), valid_set_y.reshape(-1, 1)])

    # valid_set_x = valid_set_x / 255.
    test_set_x, test_set_y = make_numpy_array(test_set)
    # test_set_x = test_set_x / 255.

    # Data normalization.
    scaler = StandardScaler()
    train_set_x = scaler.fit_transform(train_set_x)
    test_set_x = scaler.fit_transform(test_set_x)

    return [(train_set_x, train_set_y), (test_set_x, test_set_y)]


def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret


def make_numpy_array(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    # data_x = np.asarray(data_x, dtype=theano.config.floatX)
    data_x = np.asarray(data_x, dtype='float32')
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y


def trim_mnist(vdata, batch_size):
    views = []

    for i, data in enumerate(vdata):
        x_data, y_data = data

        batches = len(x_data) // batch_size

        num_samples = batches * batch_size

        x_data, y_data = x_data[:num_samples, :], y_data[:num_samples]
        views.append((x_data, y_data))

    return views


def process_evaluation_data(dloader, dim, model=None, view_idx=0):
    data_x = None
    data_y = None
    for data in dloader:
        X = data[view_idx][0]
        y = data[view_idx][1]
        if model:
            X = torch.unsqueeze(X, dim=1)
            if cuda:
                X = X.cuda()
                model = model.cuda()
            X = model(X)

            if data_x is None:
                data_x = X.cpu().detach().numpy()
                data_y = y.cpu().detach().numpy()
            else:
                data_x = np.concatenate((data_x, X.cpu().detach().numpy()), axis=0)
                data_y = np.concatenate((data_y, y.cpu().detach().numpy()), axis=0)
    return np.array(data_x).reshape(-1, dim), np.array(data_y).ravel()


def evaluate(model_tr, model_tt, ldim, tr_ldr, tt_ldr, view_idx):
    svm_train_dataset = process_evaluation_data(tr_ldr, ldim, model_tr, view_idx[0])
    svm_test_dataset = process_evaluation_data(tt_ldr, ldim, model_tt, view_idx[1])
    sanity_check, val_accuracy = svm_classify((svm_train_dataset, svm_test_dataset))
    return sanity_check, val_accuracy


def visualize(path, series, xlabel=None, ylabel=None):
    fig = plt.figure()
    legend = []
    for k in series.keys():
        plt.plot(series[k])
        legend.append(k)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend)
    plt.savefig(path)
    plt.close(fig)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", world_size=world_size, rank=rank)


def cleanup():
    dist.destroy_process_group()


def run_training(training_fn, nprocs, *args):
    mp.spawn(fn=training_fn,
             args=(nprocs, *args),
             nprocs=nprocs,
             join=True)


# def mp_train_dti(rank, nprocs, model, *args, **kwargs):
#     setup(rank, nprocs)
#     model = DDP(model)
#     res = train_dti(model, *args, **kwargs)
#     cleanup()
#     return res

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path, name):
    """
    Saves the model parameters.

    :param model:
    :param path:
    :param name:
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path=path)
    file = os.path.join(path, name + ".mod")
    torch.save(model.state_dict(), file)


def load_model(path, name):
    """
    Loads the parameters of a model.

    :param path:
    :param name:
    :return: The saved state_dict.
    """
    return torch.load(os.path.join(path, name), map_location="cuda:0")
