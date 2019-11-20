# Author: bbrighttaer
# Project: ivpgan
# Date: 7/22/19
# Time: 10:47 AM
# File: train_joint.py


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
from ivpgan.nn.models import GraphConvSequential, PairSequential, create_fcn_layers
from ivpgan.utils import Trainer, io
from ivpgan.utils.args import FcnArgs
from ivpgan.utils.sim_data import DataNode
from ivpgan.utils.train_helpers import save_model, count_parameters, load_model

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

seeds = [123, 124, 125]

check_data = False

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
# cuda = torch.cuda.is_available()
torch.cuda.set_device(0)


def create_integrated_net(hparams):
    # segment 1 - graphconv
    gconv_model = GraphConvSequential(GraphConvLayer(in_dim=75, out_dim=64),
                                      nn.BatchNorm1d(64),
                                      nn.ReLU(),
                                      GraphPool(),

                                      GraphConvLayer(in_dim=64, out_dim=64),
                                      nn.BatchNorm1d(64),
                                      nn.ReLU(),
                                      GraphPool(),

                                      nn.Linear(in_features=64, out_features=hparams["gconv_dim"]),
                                      nn.BatchNorm1d(hparams["gconv_dim"]),
                                      nn.ReLU(),
                                      nn.Dropout(hparams["dprob"]),
                                      GraphGather())

    # segment 2 - fingerprint
    fp_net = nn.Identity()

    # segment 3 - protein
    prot_net = nn.Identity()

    civ_net = PairSequential((PairSequential(mod1=(gconv_model,),
                                             mod2=(fp_net,)),),
                             (prot_net,))

    civ_dim = hparams["prot_dim"] + hparams["gconv_dim"] * 2 + hparams["fp_dim"]
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
    fcn_layers = create_fcn_layers(fcn_args)
    model = nn.Sequential(civ_net, *fcn_layers)
    return model


def create_discriminator_net(hparams):
    fcn_args = []
    p = hparams["neigh_dist"]
    layers = hparams["disc_hdims"]
    if not isinstance(layers, list):
        layers = [layers]
    for dim in layers:
        conf = FcnArgs(in_features=p,
                       out_features=dim,
                       activation='relu',
                       batch_norm=True,
                       dropout=hparams["disc_dprob"])
        fcn_args.append(conf)
        p = dim
    fcn_args.append(FcnArgs(in_features=p, out_features=1, activation="sigmoid"))
    layers = create_fcn_layers(fcn_args)
    model = nn.Sequential(*layers)
    return model


class IntegratedViewDTI(Trainer):

    @staticmethod
    def initialize(hparams, train_dataset, val_dataset, test_dataset, cuda_devices=None, mode="regression"):

        # create networks
        generator = create_integrated_net(hparams)
        discriminator = create_discriminator_net(hparams)
        print("Number of trainable parameters: generator={}, discriminator={}".format(count_parameters(generator),
                                                                                      count_parameters(discriminator)))
        if cuda:
            generator = generator.cuda()
            discriminator = discriminator.cuda()

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

        # filter optimizer arguments
        optimizer_disc = optimizer_gen = None
        for suffix in ["_gen", "_disc"]:
            key = "optimizer{}".format(suffix)

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
            }.get(hparams[key].lower(), None)
            assert optimizer is not None, "{} optimizer could not be found"

            optim_kwargs = dict()
            optim_key = hparams[key]
            for k, v in hparams.items():
                if "optimizer{}__".format(suffix) in k:
                    attribute_tup = k.split("__")
                    if optim_key == attribute_tup[1] or attribute_tup[1] == "global":
                        optim_kwargs[attribute_tup[2]] = v
            if suffix == "_gen":
                optimizer_gen = optimizer(generator.parameters(), **optim_kwargs)
            else:
                optimizer_disc = optimizer(discriminator.parameters(), **optim_kwargs)

        # metrics
        metrics = [mt.Metric(mt.rms_score, np.nanmean),
                   mt.Metric(mt.concordance_index, np.nanmean),
                   mt.Metric(mt.pearson_r2_score, np.nanmean)]
        return (generator, discriminator), (optimizer_gen, optimizer_disc), \
               {"train": train_data_loader,
                "val": val_data_loader,
                "test": test_data_loader}, metrics, hparams["weighted_loss"], hparams["neigh_dist"]

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
    def train(models, optimizers, data_loaders, metrics, weighted_loss, neigh_dist, transformers_dict,
              prot_desc_dict, tasks, n_iters=5000, is_hsearch=False, sim_data_node=None):
        generator, discriminator = models
        optimizer_gen, optimizer_disc = optimizers

        start = time.time()
        best_model_wts = generator.state_dict()
        best_score = -10000
        best_epoch = -1
        n_epochs = n_iters // len(data_loaders["train"])

        # learning rate decay schedulers
        scheduler_gen = sch.StepLR(optimizer_gen, step_size=40, gamma=0.01)
        scheduler_disc = sch.StepLR(optimizer_disc, step_size=40, gamma=0.01)

        # pred_loss functions
        prediction_criterion = nn.MSELoss()
        adversarial_loss = nn.BCELoss()

        # sub-nodes of sim data resource
        loss_lst = []
        train_loss_node = DataNode(label="training_loss", data=loss_lst)
        metrics_dict = {}
        metrics_node = DataNode(label="validation_metrics", data=metrics_dict)
        scores_lst = []
        scores_node = DataNode(label="validation_score", data=scores_lst)
        gen_loss_lst = []
        gen_loss_node = DataNode(label="generator_loss", data=gen_loss_lst)
        dis_loss_lst = []
        dis_loss_node = DataNode(label="discriminator_loss", data=dis_loss_lst)

        # add sim data nodes to parent node
        if sim_data_node:
            sim_data_node.data = [train_loss_node, metrics_node, scores_node, gen_loss_node, dis_loss_node]

        # Main training loop
        for epoch in range(n_epochs):
            for phase in ["train", "val" if is_hsearch else "test"]:
                if phase == "train":
                    print("Training....")
                    # Adjust the learning rate.
                    # scheduler_gen.step()
                    # Training mode
                    generator.train()
                    discriminator.train()
                else:
                    print("Validation...")
                    # Evaluation mode
                    generator.eval()

                data_size = 0.
                epoch_losses = []
                epoch_scores = []

                # Iterate through mini-batches
                i = 0
                for batch in tqdm(data_loaders[phase]):
                    batch_size, data = batch_collator(batch, prot_desc_dict, spec={"gconv": True,
                                                                                   "ecfp8": True})
                    # organize the data for each view.
                    Xs = {}
                    Ys = {}
                    Ws = {}
                    for view_name in data:
                        view_data = data[view_name]
                        if view_name == "gconv":
                            x = ((view_data[0][0], batch_size), view_data[0][1])
                            Xs["gconv"] = x
                        else:
                            Xs[view_name] = view_data[0]
                        Ys[view_name] = np.array([k for k in view_data[1]], dtype=np.float)
                        Ws[view_name] = np.array([k for k in view_data[2]], dtype=np.float)

                    optimizer_gen.zero_grad()
                    optimizer_disc.zero_grad()

                    # forward propagation
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        Ys = {k: Ys[k].astype(np.float) for k in Ys}
                        # Ensure corresponding pairs
                        for j in range(1, len(Ys.values())):
                            assert (list(Ys.values())[j - 1] == list(Ys.values())[j]).all()

                        y = Ys[list(Xs.keys())[0]]
                        w = Ws[list(Xs.keys())[0]]
                        X = ((Xs["gconv"][0], Xs["ecfp8"][0]), Xs["gconv"][1])

                        outputs = generator(X)
                        target = torch.from_numpy(y).float()
                        weights = torch.from_numpy(w).float()
                        valid = torch.ones_like(target).float()
                        fake = torch.zeros_like(target).float()
                        if cuda:
                            target = target.cuda()
                            weights = weights.cuda()
                            valid = valid.cuda()
                            fake = fake.cuda()
                        outputs = outputs * weights
                        pred_loss = prediction_criterion(outputs, target)

                    if phase == "train":

                        # GAN stuff
                        f_xx, f_yy = torch.meshgrid(outputs.squeeze(), outputs.squeeze())
                        predicted_diffs = torch.abs(f_xx - f_yy).sort(dim=1)[0][:, : neigh_dist]
                        r_xx, r_yy = torch.meshgrid(target.squeeze(), target.squeeze())
                        real_diffs = torch.abs(r_xx - r_yy).sort(dim=1)[0][:, :neigh_dist]

                        # generator
                        gen_loss = adversarial_loss(discriminator(predicted_diffs), valid)
                        gen_loss_lst.append(gen_loss.item())
                        loss = pred_loss + weighted_loss * gen_loss
                        loss.backward()
                        optimizer_gen.step()

                        # discriminator
                        true_loss = adversarial_loss(discriminator(real_diffs), valid)
                        fake_loss = adversarial_loss(discriminator(predicted_diffs.detach()), fake)
                        discriminator_loss = (true_loss + fake_loss) / 2.
                        dis_loss_lst.append(discriminator_loss.item())
                        discriminator_loss.backward()
                        optimizer_disc.step()

                        # for epoch stats
                        epoch_losses.append(pred_loss.item())

                        # for sim data resource
                        loss_lst.append(pred_loss.item())

                        print("\tEpoch={}/{}, batch={}/{}, pred_loss={:.4f}, D loss={:.4f}, G loss={:.4f}".format(
                            epoch + 1, n_epochs,
                            i + 1,
                            len(data_loaders[phase]),
                            pred_loss.item(),
                            discriminator_loss,
                            gen_loss))
                    else:
                        if str(pred_loss.item()) != "nan":  # useful in hyperparameter search
                            eval_dict = {}
                            score = IntegratedViewDTI.evaluate(eval_dict, y, outputs, w, metrics, tasks,
                                                               transformers_dict["gconv"])
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
                    print("\nPhase: {}, avg task pred_loss={:.4f}, ".format(phase, np.nanmean(epoch_losses)))
                    scheduler_disc.step()
                else:
                    mean_score = np.mean(epoch_scores)
                    if best_score < mean_score:
                        best_score = mean_score
                        best_model_wts = copy.deepcopy(generator.state_dict())
                        best_epoch = epoch

        duration = time.time() - start
        print('\nModel training duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        generator.load_state_dict(best_model_wts)
        return {'model': generator, 'score': best_score, 'epoch': best_epoch}

    @staticmethod
    def evaluate_model(model, model_dir, model_name, data_loaders, metrics, transformers_dict, prot_desc_dict,
                       tasks, sim_data_node=None):
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
                    batch_size, data = batch_collator(batch, prot_desc_dict, spec={"gconv": True,
                                                                                   "ecfp8": True})

                    # organize the data for each view.
                    Xs = {}
                    Ys = {}
                    Ws = {}
                    for view_name in data:
                        view_data = data[view_name]
                        if view_name == "gconv":
                            x = ((view_data[0][0], batch_size), view_data[0][1])
                            Xs["gconv"] = x
                        else:
                            Xs[view_name] = view_data[0]
                        Ys[view_name] = np.array([k for k in view_data[1]], dtype=np.float)
                        Ws[view_name] = np.array([k for k in view_data[2]], dtype=np.float)

                    # forward propagation
                    with torch.set_grad_enabled(False):
                        Ys = {k: Ys[k].astype(np.float) for k in Ys}
                        # Ensure corresponding pairs
                        for i in range(1, len(Ys.values())):
                            assert (list(Ys.values())[i - 1] == list(Ys.values())[i]).all()

                        y = Ys[list(Xs.keys())[0]]
                        w = Ws[list(Xs.keys())[0]]
                        X = ((Xs["gconv"][0], Xs["ecfp8"][0]), Xs["gconv"][1])
                        y_predicted = model(X)
                        weights = torch.from_numpy(w).float()
                        if cuda:
                            weights = weights.cuda()
                        y_predicted = y_predicted * weights

                        # apply transformers
                        predicted_vals.extend(undo_transforms(y_predicted.cpu().detach().numpy(),
                                                              transformers_dict["gconv"]).squeeze().tolist())
                        true_vals.extend(
                            undo_transforms(y, transformers_dict["gconv"]).astype(np.float).squeeze().tolist())

                    eval_dict = {}
                    score = IntegratedViewDTI.evaluate(eval_dict, y, y_predicted, w, metrics, tasks,
                                                       transformers_dict["gconv"])

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
    view = "integrated_view_gan"
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

    # For searching over multiple seeds
    hparam_search = None

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

        # Data
        data_dict["gconv"] = get_data("GraphConv", flags, prot_sequences=prot_seq_dict, seed=seed)
        transformers_dict["gconv"] = data_dict["gconv"][2]
        data_dict["ecfp8"] = get_data("ECFP8", flags, prot_sequences=prot_seq_dict, seed=seed)
        transformers_dict["ecfp8"] = data_dict["ecfp8"][2]

        tasks = data_dict["gconv"][0]

        trainer = IntegratedViewDTI()

        if flags["cv"]:
            k = flags["fold_num"]
            print("{}, {}-Prot: Training scheme: {}-fold cross-validation".format(tasks, view, k))
        else:
            k = 1
            print("{}, {}-Prot: Training scheme: train, validation".format(tasks, view)
                  + (", test split" if flags['test'] else " split"))

        if check_data:
            verify_multiview_data(data_dict)
        else:
            if flags["hparam_search"]:
                print("Hyperparameter search enabled: {}".format(flags["hparam_search_alg"]))

                # arguments to callables
                extra_init_args = {"mode": "regression",
                                   "cuda_devices": cuda_devices}
                extra_data_args = {"flags": flags,
                                   "data_dict": data_dict}
                n_iters = 3000
                extra_train_args = {"transformers_dict": transformers_dict,
                                    "prot_desc_dict": prot_desc_dict,
                                    "tasks": tasks,
                                    "n_iters": n_iters,
                                    "is_hsearch": True}

                hparams_conf = get_hparam_config(flags)

                if hparam_search is None:
                    search_alg = {"random_search": RandomSearchCV,
                                  "bayopt_search": BayesianOptSearchCV}.get(flags["hparam_search_alg"],
                                                                            BayesianOptSearchCV)
                    min_opt = "gp"
                    hparam_search = search_alg(hparam_config=hparams_conf,
                                               num_folds=k,
                                               initializer=trainer.initialize,
                                               data_provider=trainer.data_provider,
                                               train_fn=trainer.train,
                                               save_model_fn=io.save_model,
                                               init_args=extra_init_args,
                                               data_args=extra_data_args,
                                               train_args=extra_train_args,
                                               data_node=data_node,
                                               split_label=split_label,
                                               sim_label=sim_label,
                                               minimizer=min_opt,
                                               dataset_label=dataset_lbl,
                                               results_file="{}_{}_dti_{}_{}_{}.csv".format(
                                                   flags["hparam_search_alg"], sim_label, date_label, min_opt, n_iters))

                stats = hparam_search.fit(model_dir="models", model_name="".join(tasks), max_iter=20, seed=seed)
                print(stats)
                print("Best params = {}".format(stats.best(m="max")))
            else:
                invoke_train(trainer, tasks, data_dict, transformers_dict, flags, prot_desc_dict, data_node, view)

    # save simulation data resource tree to file.
    sim_data.to_json(path="./analysis/")


def invoke_train(trainer, tasks, data_dict, transformers_dict, flags, prot_desc_dict, data_node, view):
    hyper_params = default_hparams_bopt(flags)
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
    model, optimizer, data_loaders, metrics, weighted_loss, neigh_dist = trainer.initialize(hparams=hyper_params,
                                                                                            train_dataset=data["train"],
                                                                                            val_dataset=data["val"],
                                                                                            test_dataset=data["test"])
    if flags["eval"]:
        trainer.evaluate_model(model[0], flags["model_dir"], flags["eval_model_name"], data_loaders, metrics,
                               transformers_dict, prot_desc_dict, tasks, sim_data_node=sim_data_node)
    else:
        # Train the model
        results = trainer.train(model, optimizer, data_loaders, metrics, weighted_loss, neigh_dist, transformers_dict,
                                prot_desc_dict, tasks, n_iters=10000, sim_data_node=sim_data_node)
        model, score, epoch = results['model'], results['score'], results['epoch']
        # Save the model.
        split_label = "warm" if flags["split_warm"] else "cold_target" if flags["cold_target"] else "cold_drug" if \
            flags["cold_drug"] else "None"
        save_model(model, flags["model_dir"],
                   "{}_{}_{}_{}_{}_{:.4f}".format(flags["dataset"], view, flags["model_name"], split_label, epoch,
                                                  score))


def default_hparams_rand(flags):
    return {
        "prot_dim": 8421,
        "fp_dim": 1024,
        "gconv_dim": 128,
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


def default_hparams_bopt(flags):
    return {
        "prot_dim": 8421,
        "fp_dim": 1024,
        "gconv_dim": 128,
        "hdims": [2286, 1669, 2590],
        "disc_hdims": [724, 561],

        # weight initialization
        "kaiming_constant": 5,

        "weighted_loss": 0.3,

        # dropout
        "dprob": 0.0519347,
        "disc_dprob": 0.137044,

        "neigh_dist": 10,

        "tr_batch_size": 256,
        "val_batch_size": 512,
        "test_batch_size": 512,

        # optimizer params
        "optimizer_gen": "adagrad",
        "optimizer_gen__global__weight_decay": 0.00312756,
        "optimizer_gen__global__lr": 0.000867065,
        "optimizer_gen__adadelta__rho": 0.115873,
        "optimizer_gen__adagrad__lr_decay": 0.000496165,
        "optimizer_disc": "adadelta",
        "optimizer_disc__global__weight_decay": 0.0540819,
        "optimizer_disc__global__lr": 0.464296
    }


def get_hparam_config(flags):
    return {
        "prot_dim": ConstantParam(8421),
        "fp_dim": ConstantParam(1024),
        "gconv_dim": ConstantParam(128),
        "hdims": ConstantParam([2286, 1669, 2590]),
        "disc_hdims": DiscreteParam(min=100, max=1000, size=DiscreteParam(min=1, max=2)),

        # weight initialization
        "kaiming_constant": ConstantParam(5),

        "weighted_loss": LogRealParam(min=-1),

        # dropout
        "dprob": ConstantParam(0.0519347),
        "disc_dprob": LogRealParam(min=-2),
        "neigh_dist": DiscreteParam(min=5, max=20),

        "tr_batch_size": ConstantParam(256),
        "val_batch_size": ConstantParam(512),
        "test_batch_size": ConstantParam(512),

        # optimizer params
        "optimizer_gen": ConstantParam("adagrad"),
        "optimizer_gen__global__weight_decay": ConstantParam(0.00312756),
        "optimizer_gen__global__lr": ConstantParam(0.000867065),
        "optimizer_gen__adagrad__lr_decay": ConstantParam(0.000496165),
        "optimizer_disc": CategoricalParam(choices=["sgd", "adam", "adadelta", "adagrad", "adamax", "rmsprop"]),
        "optimizer_disc__global__weight_decay": LogRealParam(),
        "optimizer_disc__global__lr": LogRealParam(),

    }


def verify_multiview_data(data_dict):
    ecfp8_data = data_dict["ecfp8"][1][0]
    gconv_data = data_dict["gconv"][1][0]
    corr = []
    for i in range(100):
        print("-" * 100)
        ecfp8 = "mol={}, prot={}, y={}".format(ecfp8_data.X[i][0].smiles, ecfp8_data.X[i][1].get_name(),
                                               ecfp8_data.y[i])
        print("ecfp8:", ecfp8)
        gconv = "mol={}, prot={}, y={}".format(gconv_data.X[i][0].smiles, gconv_data.X[i][1].get_name(),
                                               gconv_data.y[i])
        print("gconv:", gconv)
        print('#' * 100)
        corr.append(ecfp8 == gconv)
    print(corr)


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
    FLAGS["eval"] = args.eval
    FLAGS["eval_model_name"] = args.eval_model_name

    main(flags=FLAGS)
