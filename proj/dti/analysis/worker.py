# Author: bbrighttaer
# Project: ivpgan
# Date: 7/17/19
# Time: 6:00 PM
# File: worker.py


from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


def get_resources(root, queries):
    """Retrieves a list of resources under a root."""
    q_res = []
    for p in queries:
        res = get_resource(p, root)
        q_res.append(res)
    return q_res


def get_resource(p, root):
    """Retrieves a single resource under a root."""
    els = p.split('/')
    els.reverse()
    res = finder(root, els)
    return res


def finder(res_tree, nodes):
    """
    Uses recursion to locate a leave resource.

    :param res_tree: The (sub)resource containing the desired leave resource.
    :param nodes: A list of nodes leading to the resource.
    :return: Located resource content.
    """
    if len(nodes) == 1:
        return res_tree[nodes[0]]
    else:
        cur_node = nodes.pop()
        try:
            return finder(res_tree[cur_node], nodes)
        except TypeError:
            return finder(res_tree[int(cur_node)], nodes)


def retrieve_resource_cv(k, seeds, r_name, r_data, res_names):
    """
    Aggregates cross-validation data for analysis.

    :param k: number of folds.
    :param seeds: A list of seeds used for the simulation.
    :param r_name: The name of the root resource.
    :param r_data: The json data.
    :param res_names: A list resource(s) under each fold to be retrieved.
                      Each record is a tuple of (leave resource path, index of resource path under the given CV fold)
    :return: A dict of the aggregated resources across seeds and folds.
    """
    query_results = dict()
    for res, idx in res_names:
        query_results[res] = []
        for i, seed in enumerate(seeds):
            for j in range(k):
                path = "{}/{}/seed_{}cv/{}/fold-{}/{}/{}".format(r_name, i, seed, j, j, idx, res)
                r = get_resource(path, r_data)
                query_results[res].append(r)
    return {k: np.array(query_results[k]) for k in query_results}


if __name__ == '__main__':
    chart_type = "png"
    folder = "eval_2019_08_16"
    qualifier = "eval_2019_08_16"
    files = [f for f in os.listdir(folder) if qualifier in f and ".json" in f]
    files.sort()
    results_folder = "results_" + folder + '_' + chart_type
    os.makedirs(results_folder, exist_ok=True)
    for file in files:
        sns.set_style("darkgrid")

        print(file)
        with open(os.path.join(folder, file), "r") as f:
            data = json.load(f)

        root_name = file.split(".j")[0]
        data_dict = retrieve_resource_cv(k=5, seeds=[123, 124, 125], r_name=root_name, r_data=data,
                                         res_names=[
                                             ("validation_metrics/nanmean-rms_score", 0),
                                             ("validation_metrics/nanmean-concordance_index", 0),
                                             ("validation_metrics/nanmean-pearson_r2_score", 0),
                                             ("validation_score", 1),
                                             ("model_predictions/y", 2),
                                             ("model_predictions/y_pred", 2)
                                         ])
        for k in data_dict:
            print('\t', k, data_dict[k].shape)
        print()

        # calculate avg rms
        rms_mean = data_dict["validation_metrics/nanmean-rms_score"].mean()
        rms_std = data_dict["validation_metrics/nanmean-rms_score"].std()
        rms_mean_std = "RMSE: mean={:.4f}, std={:.3f}".format(rms_mean, rms_std)
        print('\t', rms_mean_std)

        # calculate avg ci
        ci_mean = data_dict["validation_metrics/nanmean-concordance_index"].mean()
        ci_std = data_dict["validation_metrics/nanmean-concordance_index"].std()
        ci_mean_std = "CI: mean={:.4f}, std={:.3f}".format(ci_mean, ci_std)
        print('\t', ci_mean_std)

        # calculate avg r2
        r2_mean = data_dict["validation_metrics/nanmean-pearson_r2_score"].mean()
        r2_std = data_dict["validation_metrics/nanmean-pearson_r2_score"].std()
        r2_mean_std = "R2: mean={:.4f}, std={:.3f}".format(r2_mean, r2_std)
        print('\t', r2_mean_std)

        with open(os.path.join(results_folder, root_name + '.txt'), "w") as txt_file:
            txt_file.writelines([rms_mean_std + '\n', ci_mean_std + '\n', r2_mean_std])

        # plot and save prediction and joint plots for this root to file (w.r.t data set).
        fig, ax = plt.subplots()
        y_true = data_dict["model_predictions/y"][0]  # we select one of the predictions
        y_pred = data_dict["model_predictions/y_pred"][0]
        # for i, (y1, y2) in enumerate(zip(y_true, y_pred)):
        data = pd.DataFrame({"true value": y_true,
                             "predicted value": y_pred})
        sns.relplot(x="true value", y="true value", ax=ax, data=data, kind='line', color='r') \
            # .set_axis_labels(
        # "predicted value",
        # "true value")
        f1 = sns.relplot(x="predicted value", y="true value", ax=ax, data=data)
        # f1.set_axis_labels("predicted value", "true value")
        # f1.set(xlabel="predicted value", ylabel="true value")
        fig.savefig("./{}/{}_true-vs-pred.{}".format(results_folder, root_name, chart_type))
        plt.close(f1.fig)

        sns.set_style("white")
        f2 = sns.jointplot(x="predicted value", y="true value", data=data, kind='kde')  # , stat_func=pearsonr)
        # f2.annotate(pearsonr)
        # f2.set_axis_labels("predicted value", "true value")
        f2.savefig("./{}/{}_joint.{}".format(results_folder, root_name, chart_type))
        plt.close(f2.fig)

        print('-' * 100)
