# Modified by: bbrighttaer
# For project: ivpgan
# Date: 6/24/19
# Time: 11:02 AM
# File: evaluate.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from deepchem.trans import undo_transforms


def compute_model_performance(metrics, y_pred, y, w, transformers, tasks, n_classes=2, per_task_metrics=False):
    """
    Computes statistics of a model based and saves results to csv.

    :param metrics: list
        List of :Metric objects.
    :param y_pred: ndarray
        The predicted values.
    :param y: ndarray
        The ground truths.
    :param w: ndarray
        Label weights.
    :param transformers: list
        DeepChem/PADME data transformers used in the loading pipeline.
    :param n_classes: int, optional
        Number of classes in the data (for classification tasks only).
    :param per_task_metrics: bool, optional
        If true, return computed metric for each task on multitask dataset.
    :return:
    """

    if not len(metrics):
        return {}
    multitask_scores = {}
    all_task_scores = {}

    y = undo_transforms(y, transformers)
    y_pred = undo_transforms(y_pred, transformers)
    if len(w) != 0:
        w = np.array(w)
        w = np.reshape(w, newshape=y.shape)

    # Compute multitask metrics
    for metric in metrics:
        if per_task_metrics:
            multitask_scores[metric.name], computed_metrics = metric.compute_metric(
                y, y_pred, w, per_task_metrics=True, n_classes=n_classes, tasks=tasks)
            all_task_scores[metric.name] = computed_metrics
        else:
            multitask_scores[metric.name] = metric.compute_metric(
                y, y_pred, w, per_task_metrics=False, n_classes=n_classes, tasks=tasks)

    if not per_task_metrics:
        return multitask_scores
    else:
        return multitask_scores, all_task_scores
