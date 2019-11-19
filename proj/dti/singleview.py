# Author: bbrighttaer
# Project: ivpgan
# Date: 7/2/19
# Time: 1:24 PM
# File: singleview.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import copy
import random
import time
from datetime import datetime as dt

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as sch
from deepchem.trans import undo_transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import ivpgan.metrics as mt
from ivpgan import cuda
from ivpgan.data import batch_collator, get_data, load_proteins, DtiDataset
from soek.bopt import BayesianOptSearchCV
from soek.params import ConstantParam, LogRealParam, DiscreteParam, CategoricalParam
from soek.rand import RandomSearchCV
from ivpgan.metrics import compute_model_performance
from ivpgan.nn.layers import GraphConvLayer, GraphPool, GraphGather
from ivpgan.nn.models import create_fcn_layers, CIV, WeaveModel, GraphConvSequential, PairSequential
from ivpgan.utils import Trainer, io
from ivpgan.utils.args import FcnArgs, WeaveLayerArgs, WeaveGatherArgs
from ivpgan.utils.sim_data import DataNode
from ivpgan.utils.train_helpers import save_model, count_parameters, load_model

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

seeds = [123, 124, 125]


# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
# cuda = torch.cuda.is_available()
torch.cuda.set_device(0)


def create_ecfp_net(hparams):
    civ_dim = hparams["prot_dim"] + hparams["comp_dim"]
    fcn_args = []
    p = civ_dim
    layers = hparams["hdims"]
    if not isinstance(layers, list):
        layers = [layers]
    for dim in layers:
        conf = FcnArgs(in_features=p,
                       out_features=dim,
                       activation='relu',
                       batch_norm=True,
                       dropout=hparams["dprob"])
        fcn_args.append(conf)
        p = dim
    fcn_args.append(FcnArgs(in_features=p, out_features=1))
    layers = [CIV(dim=1)] + create_fcn_layers(fcn_args)
    model = nn.Sequential(*layers)
    return model


def create_weave_net(hparams):
    weave_args = (
        WeaveLayerArgs(n_atom_input_feat=75,
                       n_pair_input_feat=14,
                       n_atom_output_feat=50,
                       n_pair_output_feat=50,
                       n_hidden_AA=50,
                       n_hidden_PA=50,
                       n_hidden_AP=50,
                       n_hidden_PP=50,
                       update_pair=True,
                       activation='relu',
                       batch_norm=True,
                       dropout=hparams["dprob"]
                       ),
        WeaveLayerArgs(n_atom_input_feat=50,
                       n_pair_input_feat=50,
                       n_atom_output_feat=50,
                       n_pair_output_feat=50,
                       n_hidden_AA=50,
                       n_hidden_PA=50,
                       n_hidden_AP=50,
                       n_hidden_PP=50,
                       update_pair=True,
                       batch_norm=True,
                       dropout=hparams["dprob"],
                       activation='relu'),
    )
    wg_args = WeaveGatherArgs(conv_out_depth=50, gaussian_expand=True, n_depth=128)
    weave_model = WeaveModel(weave_args, wg_args)

    # FCN
    civ_dim = hparams["prot_dim"] + 128
    fcn_args = []
    p = civ_dim
    fcn_layers = hparams["hdims"]
    if not isinstance(fcn_layers, list):
        fcn_layers = [fcn_layers]
    for dim in fcn_layers:
        conf = FcnArgs(in_features=p,
                       out_features=dim,
                       activation='relu',
                       batch_norm=True,
                       dropout=hparams["dprob"])
        fcn_args.append(conf)
        p = dim
    fcn_args.append(FcnArgs(in_features=p, out_features=1))
    fcn_layers = create_fcn_layers(fcn_args)

    model = nn.Sequential(PairSequential(mod1=(weave_model,),
                                         mod2=(nn.Identity(),)),
                          *fcn_layers)
    return model


def create_gconv_net(hparams):
    gconv_model = GraphConvSequential(GraphConvLayer(in_dim=75, out_dim=64),
                                      nn.BatchNorm1d(64),
                                      nn.ReLU(),
                                      GraphPool(),

                                      GraphConvLayer(in_dim=64, out_dim=64),
                                      nn.BatchNorm1d(64),
                                      nn.ReLU(),
                                      GraphPool(),

                                      nn.Linear(in_features=64, out_features=128),
                                      nn.BatchNorm1d(128),
                                      nn.ReLU(),
                                      nn.Dropout(hparams["dprob"]),
                                      GraphGather())
    # FCN
    civ_dim = hparams["prot_dim"] + 128 * 2
    fcn_args = []
    p = civ_dim
    fcn_layers = hparams["hdims"]
    if not isinstance(fcn_layers, list):
        fcn_layers = [fcn_layers]
    for dim in fcn_layers:
        conf = FcnArgs(in_features=p,
                       out_features=dim,
                       activation='relu',
                       batch_norm=True,
                       dropout=hparams["dprob"])
        fcn_args.append(conf)
        p = dim
    fcn_args.append(FcnArgs(in_features=p, out_features=1))
    fcn_layers = create_fcn_layers(fcn_args)

    model = nn.Sequential(PairSequential(mod1=(gconv_model,),
                                         mod2=(nn.Identity(),)),
                          *fcn_layers)

    return model


class SingleViewDTI(Trainer):

    @staticmethod
    def initialize(hparams, train_dataset, val_dataset, test_dataset, cuda_devices=None, mode="regression"):

        # create network
        create_func = {"ecfp4": create_ecfp_net,
                       "ecfp8": create_ecfp_net,
                       "weave": create_weave_net,
                       "gconv": create_gconv_net}.get(hparams["view"])
        model = create_func(hparams)
        print("Number of trainable parameters = {}".format(count_parameters(model)))
        if cuda:
            model = model.cuda()

        # data loaders
        train_data_loader = DataLoader(dataset=train_dataset,
                                       batch_size=hparams["tr_batch_size"],
                                       shuffle=True,
                                       collate_fn=lambda x: x)
        val_data_loader = DataLoader(dataset=val_dataset,
                                     batch_size=hparams["val_batch_size"],
                                     shuffle=False,
                                     collate_fn=lambda x: x)
        test_data_loader = None
        if test_dataset is not None:
            test_data_loader = DataLoader(dataset=test_dataset,
                                          batch_size=hparams["test_batch_size"],
                                          shuffle=False,
                                          collate_fn=lambda x: x)

        # optimizer configuration
        optimizer = {
            "adadelta": torch.optim.Adadelta,
            "adagrad": torch.optim.Adagrad,
            "adam": torch.optim.Adam,
            "adamax": torch.optim.Adamax,
            "asgd": torch.optim.ASGD,
            "rmsprop": torch.optim.RMSprop,
            "Rprop": torch.optim.Rprop,
            "sgd": torch.optim.SGD,
        }.get(hparams["optimizer"].lower(), None)
        assert optimizer is not None, "{} optimizer could not be found"

        # filter optimizer arguments
        optim_kwargs = dict()
        optim_key = hparams["optimizer"]
        for k, v in hparams.items():
            if "optimizer__" in k:
                attribute_tup = k.split("__")
                if optim_key == attribute_tup[1] or attribute_tup[1] == "global":
                    optim_kwargs[attribute_tup[2]] = v
        optimizer = optimizer(model.parameters(), **optim_kwargs)

        # metrics
        metrics = [mt.Metric(mt.rms_score, np.nanmean),
                   mt.Metric(mt.concordance_index, np.nanmean),
                   mt.Metric(mt.pearson_r2_score, np.nanmean)]
        return model, optimizer, {"train": train_data_loader,
                                  "val": val_data_loader,
                                  "test": test_data_loader}, metrics

    @staticmethod
    def data_provider(fold, flags, data_dict):
        if not flags['cv']:
            print("Training scheme: train, validation" + (", test split" if flags['test'] else " split"))
            train_dataset = DtiDataset(x_s=[data[1][0].X for data in data_dict.values()],
                                       y_s=[data[1][0].y for data in data_dict.values()],
                                       w_s=[data[1][0].w for data in data_dict.values()])
            valid_dataset = DtiDataset(x_s=[data[1][1].X for data in data_dict.values()],
                                       y_s=[data[1][1].y for data in data_dict.values()],
                                       w_s=[data[1][1].w for data in data_dict.values()])
            test_dataset = None
            if flags['test']:
                test_dataset = DtiDataset(x_s=[data[1][2].X for data in data_dict.values()],
                                          y_s=[data[1][2].y for data in data_dict.values()],
                                          w_s=[data[1][2].w for data in data_dict.values()])
            data = {"train": train_dataset, "val": valid_dataset, "test": test_dataset}
        else:
            train_dataset = DtiDataset(x_s=[data[1][fold][0].X for data in data_dict.values()],
                                       y_s=[data[1][fold][0].y for data in data_dict.values()],
                                       w_s=[data[1][fold][0].w for data in data_dict.values()])
            valid_dataset = DtiDataset(x_s=[data[1][fold][1].X for data in data_dict.values()],
                                       y_s=[data[1][fold][1].y for data in data_dict.values()],
                                       w_s=[data[1][fold][1].w for data in data_dict.values()])
            test_dataset = DtiDataset(x_s=[data[1][fold][2].X for data in data_dict.values()],
                                      y_s=[data[1][fold][2].y for data in data_dict.values()],
                                      w_s=[data[1][fold][2].w for data in data_dict.values()])
            data = {"train": train_dataset, "val": valid_dataset, "test": test_dataset}
        return data

    @staticmethod
    def evaluate(eval_dict, y, y_pred, w, metrics, tasks, transformers):
        y = y.reshape(-1, 1).astype(np.float)
        eval_dict.update(compute_model_performance(metrics, y_pred.cpu().detach().numpy(), y, w, transformers,
                                                   tasks=tasks))
        # scoring
        rms = np.nanmean(eval_dict["nanmean-rms_score"])
        ci = np.nanmean(eval_dict["nanmean-concordance_index"])
        r2 = np.nanmean(eval_dict["nanmean-pearson_r2_score"])
        score = np.nanmean([ci, r2]) - rms
        return score

    @staticmethod
    def train(eval_fn, model, optimizer, data_loaders, metrics, transformers_dict, prot_desc_dict, tasks, view,
              n_iters=5000, is_hsearch=False, sim_data_node=None):
        start = time.time()
        best_model_wts = model.state_dict()
        best_score = -10000
        best_epoch = -1
        n_epochs = n_iters // len(data_loaders["train"])
        scheduler = sch.StepLR(optimizer, step_size=40, gamma=0.01)
        criterion = torch.nn.MSELoss()

        # sub-nodes of sim data resource
        loss_lst = []
        train_loss_node = DataNode(label="training_loss", data=loss_lst)
        metrics_dict = {}
        metrics_node = DataNode(label="validation_metrics", data=metrics_dict)
        scores_lst = []
        scores_node = DataNode(label="validation_score", data=scores_lst)

        # add sim data nodes to parent node
        if sim_data_node:
            sim_data_node.data = [train_loss_node, metrics_node, scores_node]

        # Main training loop
        for epoch in range(n_epochs):
            for phase in ["train", "val" if is_hsearch else "test"]:
                if phase == "train":
                    print("Training....")
                    # Training mode
                    model.train()
                else:
                    print("Validation...")
                    # Evaluation mode
                    model.eval()

                data_size = 0.
                epoch_losses = []
                epoch_scores = []

                # Iterate through mini-batches
                i = 0
                for batch in tqdm(data_loaders[phase]):
                    batch_size, data = batch_collator(batch, prot_desc_dict, spec=view)
                    # Data
                    if view == "gconv":
                        # graph data structure is: [(compound data, batch_size), protein_data]
                        X = ((data[view][0][0], batch_size), data[view][0][1])
                    else:
                        X = data[view][0]
                    y = data[view][1]
                    w = data[view][2].reshape(-1, 1).astype(np.float)

                    optimizer.zero_grad()

                    # forward propagation
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(X)
                        target = torch.from_numpy(y.astype(np.float)).view(-1, 1).float()
                        if cuda:
                            target = target.cuda()
                        loss = criterion(outputs, target)

                    if phase == "train":
                        print("\tEpoch={}/{}, batch={}/{}, loss={:.4f}".format(epoch + 1, n_epochs, i + 1,
                                                                               len(data_loaders[phase]), loss.item()))
                        # for epoch stats
                        epoch_losses.append(loss.item())

                        # for sim data resource
                        loss_lst.append(loss.item())

                        # optimization ops
                        loss.backward()
                        optimizer.step()
                    else:
                        if str(loss.item()) != "nan":  # useful in hyperparameter search
                            eval_dict = {}
                            score = eval_fn(eval_dict, y, outputs, w, metrics, tasks, transformers_dict[view])
                            # for epoch stats
                            epoch_scores.append(score)

                            # for sim data resource
                            scores_lst.append(score)
                            for m in eval_dict:
                                if m in metrics_dict:
                                    metrics_dict[m].append(eval_dict[m])
                                else:
                                    metrics_dict[m] = [eval_dict[m]]

                            print("\nEpoch={}/{}, batch={}/{}, "
                                  "evaluation results= {}, score={}".format(epoch + 1, n_epochs, i + 1,
                                                                            len(data_loaders[phase]),
                                                                            eval_dict, score))

                    i += 1
                    data_size += batch_size
                # End of mini=batch iterations.

                if phase == "train":
                    # Adjust the learning rate.
                    scheduler.step()
                    print("\nPhase: {}, avg task loss={:.4f}, ".format(phase, np.nanmean(epoch_losses)))
                else:
                    mean_score = np.mean(epoch_scores)
                    if best_score < mean_score:
                        best_score = mean_score
                        best_model_wts = copy.deepcopy(model.state_dict())
                        best_epoch = epoch

        duration = time.time() - start
        print('\nModel training duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        model.load_state_dict(best_model_wts)
        return model, best_score, best_epoch

    @staticmethod
    def evaluate_model(eval_fn, model, model_dir, model_name, data_loaders, metrics, transformers_dict, prot_desc_dict,
                       tasks, view, sim_data_node=None):
        # load saved model and put in evaluation mode
        model.load_state_dict(load_model(model_dir, model_name))
        model.eval()

        print("Model evaluation...")
        start = time.time()
        n_epochs = 1

        # sub-nodes of sim data resource
        # loss_lst = []
        # train_loss_node = DataNode(label="training_loss", data=loss_lst)
        metrics_dict = {}
        metrics_node = DataNode(label="validation_metrics", data=metrics_dict)
        scores_lst = []
        scores_node = DataNode(label="validation_score", data=scores_lst)
        predicted_vals = []
        true_vals = []
        model_preds_node = DataNode(label="model_predictions", data={"y": true_vals,
                                                                     "y_pred": predicted_vals})

        # add sim data nodes to parent node
        if sim_data_node:
            sim_data_node.data = [metrics_node, scores_node, model_preds_node]

        # Main evaluation loop
        for epoch in range(n_epochs):

            for phase in ["test"]:  # ["train", "val"]:
                # Iterate through mini-batches
                i = 0
                for batch in tqdm(data_loaders[phase]):
                    batch_size, data = batch_collator(batch, prot_desc_dict, spec=view)
                    # Data
                    if view == "gconv":
                        # graph data structure is: [(compound data, batch_size), protein_data]
                        X = ((data[view][0][0], batch_size), data[view][0][1])
                    else:
                        X = data[view][0]
                    y_true = data[view][1]
                    w = data[view][2].reshape(-1, 1).astype(np.float)

                    # forward propagation
                    with torch.set_grad_enabled(False):
                        y_predicted = model(X)

                        # apply transformers
                        predicted_vals.extend(undo_transforms(y_predicted.cpu().detach().numpy(),
                                                              transformers_dict[view]).squeeze().tolist())
                        true_vals.extend(undo_transforms(y_true,
                                                         transformers_dict[view]).astype(np.float).squeeze().tolist())

                    eval_dict = {}
                    score = eval_fn(eval_dict, y_true, y_predicted, w, metrics, tasks, transformers_dict[view])

                    # for sim data resource
                    scores_lst.append(score)
                    for m in eval_dict:
                        if m in metrics_dict:
                            metrics_dict[m].append(eval_dict[m])
                        else:
                            metrics_dict[m] = [eval_dict[m]]

                    print("\nEpoch={}/{}, batch={}/{}, "
                          "evaluation results= {}, score={}".format(epoch + 1, n_epochs, i + 1,
                                                                    len(data_loaders[phase]),
                                                                    eval_dict, score))

                    i += 1
                # End of mini=batch iterations.

        duration = time.time() - start
        print('\nModel evaluation duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))


def main(flags):
    if len(flags["views"]) > 0:
        print("Single views for training:", flags["views"])
    else:
        print("No views selected for training")

    for view in flags["views"]:
        sim_label = "CUDA={}, view={}".format(cuda, view)
        print(sim_label)

        # Simulation data resource tree
        split_label = "warm" if flags["split_warm"] else "cold_target" if flags["cold_target"] else "cold_drug" if \
            flags["cold_drug"] else "None"
        dataset_lbl = flags["dataset"]
        node_label = "{}_{}_{}_{}_{}".format(dataset_lbl, view, split_label, "eval" if flags["eval"] else "train",
                                             date_label)
        sim_data = DataNode(label=node_label)
        nodes_list = []
        sim_data.data = nodes_list

        num_cuda_dvcs = torch.cuda.device_count()
        cuda_devices = None if num_cuda_dvcs == 1 else [i for i in range(1, num_cuda_dvcs)]

        prot_desc_dict, prot_seq_dict = load_proteins(flags['prot_desc_path'])

        for seed in seeds:
            # for data collection of this round of simulation.
            data_node = DataNode(label="seed_%d" % seed)
            nodes_list.append(data_node)

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # load data
            print('-------------------------------------')
            print('Running on dataset: %s' % dataset_lbl)
            print('-------------------------------------')

            data_dict = dict()
            transformers_dict = dict()
            data_key = {"ecfp4": "ECFP4",
                        "ecfp8": "ECFP8",
                        "weave": "Weave",
                        "gconv": "GraphConv"}.get(view)
            data_dict[view] = get_data(data_key, flags, prot_sequences=prot_seq_dict, seed=seed)
            transformers_dict[view] = data_dict[view][2]

            tasks = data_dict[view][0]

            trainer = SingleViewDTI()

            if flags["cv"]:
                k = flags["fold_num"]
                print("{}, {}-Prot: Training scheme: {}-fold cross-validation".format(tasks, view, k))
            else:
                k = 1
                print("{}, {}-Prot: Training scheme: train, validation".format(tasks, view)
                      + (", test split" if flags['test'] else " split"))

            if flags["hparam_search"]:
                print("Hyperparameter search enabled: {}".format(flags["hparam_search_alg"]))

                # arguments to callables
                extra_init_args = {"mode": "regression",
                                   "cuda_devices": cuda_devices}
                extra_data_args = {"flags": flags,
                                   "data_dict": data_dict}
                extra_train_args = {"transformers_dict": transformers_dict,
                                    "prot_desc_dict": prot_desc_dict,
                                    "tasks": tasks,
                                    "is_hsearch": True,
                                    "n_iters": 10000,
                                    "view": view}

                hparams_conf = get_hparam_config(flags, view)

                search_alg = {"random_search": RandomSearchCV,
                              "bayopt_search": BayesianOptSearchCV}.get(flags["hparam_search_alg"],
                                                                        BayesianOptSearchCV)

                hparam_search = search_alg(hparam_config=hparams_conf,
                                           num_folds=k,
                                           initializer=trainer.initialize,
                                           data_provider=trainer.data_provider,
                                           train_fn=trainer.train,
                                           eval_fn=trainer.evaluate,
                                           save_model_fn=io.save_model,
                                           init_args=extra_init_args,
                                           data_args=extra_data_args,
                                           train_args=extra_train_args,
                                           data_node=data_node,
                                           split_label=split_label,
                                           sim_label=view,
                                           dataset_label=dataset_lbl,
                                           results_file="{}_{}_dti_{}.csv".format(flags["hparam_search_alg"], view,
                                                                                  date_label))

                stats = hparam_search.fit(model_dir="models", model_name="".join(tasks), max_iter=40, seed=seed)
                print(stats)
                print("Best params = {}".format(stats.best(m="max")))
            else:
                invoke_train(trainer, tasks, data_dict, transformers_dict, flags, prot_desc_dict, data_node, view)

        # save simulation data resource tree to file.
        sim_data.to_json(path="./analysis/")


def invoke_train(trainer, tasks, data_dict, transformers_dict, flags, prot_desc_dict, data_node, view):
    hyper_params = default_hparams_bopt(flags, view)
    # Initialize the model and other related entities for training.
    if flags["cv"]:
        folds_data = []
        data_node.data = folds_data
        data_node.label = data_node.label + "cv"
        for k in range(flags["fold_num"]):
            k_node = DataNode(label="fold-%d" % k)
            folds_data.append(k_node)
            start_fold(k_node, data_dict, flags, hyper_params, prot_desc_dict, tasks, trainer,
                       transformers_dict, view, k)
    else:
        start_fold(data_node, data_dict, flags, hyper_params, prot_desc_dict, tasks, trainer,
                   transformers_dict, view)


def start_fold(sim_data_node, data_dict, flags, hyper_params, prot_desc_dict, tasks, trainer,
               transformers_dict, view, k=None):
    data = trainer.data_provider(k, flags, data_dict)
    model, optimizer, data_loaders, metrics = trainer.initialize(hparams=hyper_params,
                                                                 train_dataset=data["train"],
                                                                 val_dataset=data["val"],
                                                                 test_dataset=data["test"])
    if flags["eval"]:
        trainer.evaluate_model(trainer.evaluate, model, flags["model_dir"], flags["eval_model_name"],
                               data_loaders, metrics, transformers_dict,
                               prot_desc_dict, tasks, view=view, sim_data_node=sim_data_node)
    else:
        # Train the model
        model, score, epoch = trainer.train(trainer.evaluate, model, optimizer, data_loaders, metrics,
                                            transformers_dict,
                                            prot_desc_dict, tasks, n_iters=10000, view=view,
                                            sim_data_node=sim_data_node)
        # Save the model.
        split_label = "warm" if flags["split_warm"] else "cold_target" if flags["cold_target"] else "cold_drug" if \
            flags["cold_drug"] else "None"
        save_model(model, flags["model_dir"],
                   "{}_{}_{}_{}_{}_{:.4f}".format(flags["dataset"], view, flags["model_name"], split_label, epoch,
                                                  score))


def default_hparams_rand(flags, view):
    return {
        "view": view,
        "prot_dim": 8421,
        "comp_dim": 1024,
        "hdims": [3795, 2248, 2769, 2117],

        # weight initialization
        "kaiming_constant": 5,

        # dropout regs
        "dprob": 0.0739227,

        "tr_batch_size": 256,
        "val_batch_size": 512,
        "test_batch_size": 512,

        # optimizer params
        "optimizer": "rmsprop",
        "optimizer__sgd__weight_decay": 1e-4,
        "optimizer__sgd__nesterov": True,
        "optimizer__sgd__momentum": 0.9,
        "optimizer__sgd__lr": 1e-3,

        "optimizer__adam__weight_decay": 1e-4,
        "optimizer__adam__lr": 1e-3,

        "optimizer__rmsprop__lr": 0.000235395,
        "optimizer__rmsprop__weight_decay": 0.000146688,
        "optimizer__rmsprop__momentum": 0.00622082,
        "optimizer__rmsprop__centered": False
    }


def default_hparams_bopt(flags, view):
    return {
        "view": view,
        "prot_dim": 8421,
        "comp_dim": 1024,
        "hdims": [653, 3635],

        # weight initialization
        "kaiming_constant": 5,

        # dropout regs
        "dprob": 0.096421,

        "tr_batch_size": 256,
        "val_batch_size": 512,
        "test_batch_size": 512,

        # optimizer params
        "optimizer": "adadelta",
        "optimizer__global__weight_decay": 0.004665,
        "optimizer__global__lr": 0.04158,
        "optimizer__adadelta__rho": 0.115873,
    }


def get_hparam_config(flags, view):
    return {
        "view": ConstantParam(view),
        "prot_dim": ConstantParam(8421),
        "comp_dim": ConstantParam(1024),
        "hdims": DiscreteParam(min=256, max=5000, size=DiscreteParam(min=1, max=4)),

        # weight initialization
        "kaiming_constant": ConstantParam(5),  # DiscreteParam(min=2, max=9),

        # dropout regs
        "dprob": LogRealParam(min=-2),

        "tr_batch_size": CategoricalParam(choices=[32, 64, 128, 256, 512]),
        "val_batch_size": ConstantParam(512),
        "test_batch_size": ConstantParam(512),

        # optimizer params
        "optimizer": CategoricalParam(choices=["sgd", "adam", "adadelta", "adagrad", "adamax", "rmsprop"]),
        "optimizer__global__weight_decay": LogRealParam(),
        "optimizer__global__lr": LogRealParam(),

        # SGD
        "optimizer__sgd__nesterov": CategoricalParam(choices=[True, False]),
        "optimizer__sgd__momentum": LogRealParam(),
        # "optimizer__sgd__lr": LogRealParam(),

        # ADAM
        # "optimizer__adam__lr": LogRealParam(),
        "optimizer__adam__amsgrad": CategoricalParam(choices=[True, False]),

        # Adadelta
        # "optimizer__adadelta__lr": LogRealParam(),
        # "optimizer__adadelta__weight_decay": LogRealParam(),
        "optimizer__adadelta__rho": LogRealParam(),

        # Adagrad
        # "optimizer__adagrad__lr": LogRealParam(),
        "optimizer__adagrad__lr_decay": LogRealParam(),
        # "optimizer__adagrad__weight_decay": LogRealParam(),

        # Adamax
        # "optimizer__adamax__lr": LogRealParam(),
        # "optimizer__adamax__weight_decay": LogRealParam(),

        # RMSprop
        # "optimizer__rmsprop__lr": LogRealParam(),
        # "optimizer__rmsprop__weight_decay": LogRealParam(),
        "optimizer__rmsprop__momentum": LogRealParam(),
        # "optimizer__rmsprop__centered": CategoricalParam(choices=[True, False])

    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DTI with ivpgan model training.")

    parser.add_argument("--dataset",
                        type=str,
                        default="davis",
                        help="Dataset name.")

    # Either CV or standard train-val(-test) split.
    scheme = parser.add_mutually_exclusive_group()
    scheme.add_argument("--fold_num",
                        default=-1,
                        type=int,
                        choices=range(3, 11),
                        help="Number of folds for cross-validation")
    scheme.add_argument("--test",
                        action="store_true",
                        help="Whether a test set should be included in the data split")

    parser.add_argument("--splitting_alg",
                        choices=["random", "scaffold", "butina", "index", "task"],
                        default="random",
                        type=str,
                        help="Data splitting algorithm to use.")
    parser.add_argument('--filter_threshold',
                        type=int,
                        default=6,
                        help='Threshold such that entities with observations no more than it would be filtered out.'
                        )
    parser.add_argument('--cold_drug',
                        default=False,
                        help='Flag of whether the split will leave "cold" drugs in the test data.',
                        action='store_true'
                        )
    parser.add_argument('--cold_target',
                        default=False,
                        help='Flag of whether the split will leave "cold" targets in the test data.',
                        action='store_true'
                        )
    parser.add_argument('--cold_drug_cluster',
                        default=False,
                        help='Flag of whether the split will leave "cold cluster" drugs in the test data.',
                        action='store_true'
                        )
    parser.add_argument('--predict_cold',
                        default=False,
                        help='Flag of whether the split will leave "cold" entities in the test data.',
                        action='store_true')
    parser.add_argument('--split_warm',
                        default=False,
                        help='Flag of whether the split will not leave "cold" entities in the test data.',
                        action='store_true'
                        )
    parser.add_argument('--model_dir',
                        type=str,
                        default='./model_dir',
                        help='Directory to store the log files in the training process.'
                        )
    parser.add_argument('--model_name',
                        type=str,
                        default='model-{}'.format(date_label),
                        help='Directory to store the log files in the training process.'
                        )
    parser.add_argument('--prot_desc_path',
                        action='append',
                        help='A list containing paths to protein descriptors.'
                        )
    # parser.add_argument('--seed',
    #                     type=int,
    #                     action="append",
    #                     default=[123, 124, 125],
    #                     help='Random seeds to be used.')
    parser.add_argument('--no_reload',
                        action="store_false",
                        dest='reload',
                        help='Whether datasets will be reloaded from existing ones or newly constructed.'
                        )
    parser.add_argument('--data_dir',
                        type=str,
                        default='../../data/',
                        help='Root folder of data (Davis, KIBA, Metz) folders.')
    parser.add_argument("--hparam_search",
                        action="store_true",
                        help="If true, hyperparameter searching would be performed.")
    parser.add_argument("--hparam_search_alg",
                        type=str,
                        default="bayopt_search",
                        help="Hyperparameter search algorithm to use. One of [bayopt_search, random_search]")
    parser.add_argument("--view",
                        action="append",
                        help="The view to be simulated. One of [ecfp4, ecfp8, weave, gconv]")
    parser.add_argument("--eval",
                        action="store_true",
                        help="If true, a saved model is loaded and evaluated using CV")
    parser.add_argument("--eval_model_name",
                        default=None,
                        type=str,
                        help="The filename of the model to be loaded from the directory specified in --model_dir")

    args = parser.parse_args()

    FLAGS = dict()
    FLAGS['dataset'] = args.dataset
    FLAGS['fold_num'] = args.fold_num
    FLAGS['cv'] = True if FLAGS['fold_num'] > 2 else False
    FLAGS['test'] = args.test
    FLAGS['splitting_alg'] = args.splitting_alg
    FLAGS['filter_threshold'] = args.filter_threshold
    FLAGS['cold_drug'] = args.cold_drug
    FLAGS['cold_target'] = args.cold_target
    FLAGS['cold_drug_cluster'] = args.cold_drug_cluster
    FLAGS['predict_cold'] = args.predict_cold
    FLAGS['model_dir'] = args.model_dir
    FLAGS['model_name'] = args.model_name
    FLAGS['prot_desc_path'] = args.prot_desc_path
    # FLAGS['seeds'] = args.seed
    FLAGS['reload'] = args.reload
    FLAGS['data_dir'] = args.data_dir
    FLAGS['split_warm'] = args.split_warm
    FLAGS['hparam_search'] = args.hparam_search
    FLAGS["hparam_search_alg"] = args.hparam_search_alg
    FLAGS["views"] = args.view
    FLAGS["eval"] = args.eval
    FLAGS["eval_model_name"] = args.eval_model_name

    main(flags=FLAGS)
