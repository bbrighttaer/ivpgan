# Author: bbrighttaer
# Project: ivpgan
# Date: 5/23/19
# Time: 10:43 AM
# File: io.py


from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import pickle

import padme

__author__ = 'Brighter Agyemang'

import os
import logging
import sys
import torch


def get_logger(name=None, level='INFO', stream='stderr', filename=None, log_dir='./logs/'):
    """
    Creates and return a logger to both console and a specified file.

    :param log_dir: The directory of the log file
    :param filename: The file to be logged into. It shall be in ./logs/
    :param name: The name of the logger
    :param level: The logging level; one of DEBUG, INFO, WARNING, ERROR, CRITICAL
    :return: The created logger
    :param stream: Either 'stderr' or 'stdout'
    """
    stream = sys.stderr if stream == 'stderr' else sys.stdout
    log_level = {'DEBUG': logging.DEBUG,
                 'INFO': logging.INFO,
                 'WARNING': logging.WARNING,
                 'ERROR': logging.ERROR,
                 'CRITICAL': logging.CRITICAL}.get(level.upper(), 'INFO')
    handlers = []
    if filename:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        handlers.append(logging.FileHandler(os.path.join(log_dir, filename + '.log')))
    if stream:
        handlers.append(logging.StreamHandler(stream))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=handlers)
    return logging.getLogger(name)

def save_model(model, path, name):
    """
    Saves the model parameters.

    :param model:
    :param path:
    :param name:
    :return:
    """
    os.makedirs(path, exist_ok=True)
    file = os.path.join(path, name + ".mod")
    torch.save(model.state_dict(), file)


def load_model(path, name):
    """
    Loads the parameters of a model.

    :param path:
    :param name:
    :return: The saved state_dict.
    """
    return torch.load(os.path.join(path, name), map_location=torch.device("cpu"))

def save_nested_cv_dataset_to_disk(save_dir, fold_dataset, fold_num, transformers):
    assert fold_num > 1
    for i in range(fold_num):
        fold_dir = os.path.join(save_dir, "fold" + str(i + 1))
        train_dir = os.path.join(fold_dir, "train_dir")
        valid_dir = os.path.join(fold_dir, "valid_dir")
        test_dir = os.path.join(fold_dir, "test_dir")
        train_data = fold_dataset[i][0]
        valid_data = fold_dataset[i][1]
        test_data = fold_dataset[i][2]
        train_data.move(train_dir)
        valid_data.move(valid_dir)
        test_data.move(test_dir)
    with open(os.path.join(save_dir, "transformers.pkl"), "wb") as f:
        pickle.dump(transformers, f)
    return None


def load_nested_cv_dataset_from_disk(save_dir, fold_num):
    assert fold_num > 1
    loaded = False
    train_data = []
    valid_data = []
    test_data = []
    for i in range(fold_num):
        fold_dir = os.path.join(save_dir, "fold" + str(i + 1))
        train_dir = os.path.join(fold_dir, "train_dir")
        valid_dir = os.path.join(fold_dir, "valid_dir")
        test_dir = os.path.join(fold_dir, "test_dir")
        if not os.path.exists(train_dir) or not os.path.exists(valid_dir) or not os.path.exists(test_dir):
            return False, None, list()
        train = padme.data.DiskDataset(train_dir)
        valid = padme.data.DiskDataset(valid_dir)
        test = padme.data.DiskDataset(test_dir)
        train_data.append(train)
        valid_data.append(valid)
        test_data.append(test)

    loaded = True
    with open(os.path.join(save_dir, "transformers.pkl"), 'rb') as f:
        transformers = pickle.load(f)
        return loaded, list(zip(train_data, valid_data, test_data)), transformers