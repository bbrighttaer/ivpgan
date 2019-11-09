# Author: bbrighttaer
# Project: ivpgan
# Date: 5/23/19
# Time: 10:43 AM
# File: io.py


from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

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
