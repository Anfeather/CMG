from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import time
import logging
import argparse
from collections import OrderedDict
import faiss

import torch
import torch.nn as nn

from models import SupResNet, SSLResNet
from utils import (
    get_features,
    get_roc_sklearn,
    get_pr_sklearn,
    get_fpr,
    get_scores_one_cluster,
)
import data

# local utils for SSD evaluation
def get_scores(ftrain, ftest, food, args):
    if args.clusters == 1:
        return get_scores_one_cluster(ftrain, ftest, food)
    else:
        if args.training_mode == "SupCE":
            print("Using data labels as cluster since model is cross-entropy")
    
        else:
            ypred = get_clusters(ftrain, args.clusters)
        return get_scores_multi_cluster(ftrain, ftest, food, ypred)


def get_clusters(ftrain, nclusters):
    kmeans = faiss.Kmeans(
        ftrain.shape[1], nclusters, niter=100, verbose=False, gpu=False
    )
    kmeans.train(np.random.permutation(ftrain))
    _, ypred = kmeans.assign(ftrain)
    return ypred


def get_scores_multi_cluster(ftrain, ftest, food, ypred):
    xc = [ftrain[ypred == i] for i in np.unique(ypred)]

    din = [
        np.sum(
            (ftest - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (ftest - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]
    dood = [
        np.sum(
            (food - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (food - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]

    din = np.min(din, axis=0)
    dood = np.min(dood, axis=0)

    return din, dood


def get_eval_results(ftrain, ftest, food, args):
    """
    None.
    """
    # standardize data
    ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
    ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
    food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10

    m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)

    ftrain = (ftrain - m) / (s + 1e-10)
    ftest = (ftest - m) / (s + 1e-10)
    food = (food - m) / (s + 1e-10)

    dtest, dood = get_scores(ftrain, ftest, food,  args)

    fpr95 = get_fpr(dtest, dood)
    auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)
    return fpr95, auroc, aupr


def main():
    parser = argparse.ArgumentParser(description="SSD evaluation")

    parser.add_argument("--exp-name", type=str, default="temp_eval_ssd")
    parser.add_argument(
        "--training-mode", type=str, choices=("SimCLR", "SupCon", "SupCE")
    )
    parser.add_argument("--results-dir", type=str, default="./eval_results")

    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=1)

    parser.add_argument("--dataset", type=str, default="MSCOCO")
    parser.add_argument(
        "--data-dir", type=str, default="/home/ray/preject/data/non_iid_MSCOCO_train_30_50/"
    )
    parser.add_argument(
        "--data-mode", type=str, choices=("org", "base", "ssl"), default="base"
    )
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--size", type=int, default=32)

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--ckpt", type=str, help="checkpoint path")
    parser.add_argument("--seed", type=int, default=12345)

    args = parser.parse_args()
    device = "cuda:0"
    #40 40
    #user_dict = {0:[67, 69, 33, 72, 49, 2, 55, 28, 76, 54, 5, 3, 71, 46, 50, 59, 7, 0, 16, 53, 77, 13, 15, 10, 8, 9, 75, 11, 17, 21, 18, 19, 61, 20, 6, 14, 22, 74, 4, 23],1:[24, 35, 78, 26, 42, 73, 32, 38, 64, 34, 70, 44, 57, 40, 30, 31, 41, 45, 29, 60, 56, 43, 58, 36, 39, 66, 63, 37, 27, 62, 51, 79, 65, 68, 47, 1, 12, 25, 52, 48]}
    #30 50
    user_dict = {0:[ 5, 3, 71, 46, 50, 59, 7, 0, 16, 53, 77, 13, 15, 10, 8, 9, 75, 11, 17, 21, 18, 19, 61, 20, 6, 14, 22, 74, 4, 23],1:[67, 69, 33, 72, 49, 2, 55, 28, 76, 54, 24, 35, 78, 26, 42, 73, 32, 38, 64, 34, 70, 44, 57, 40, 30, 31, 41, 45, 29, 60, 56, 43, 58, 36, 39, 66, 63, 37, 27, 62, 51, 79, 65, 68, 47, 1, 12, 25, 52, 48]}

    assert args.ckpt, "Must provide a checkpint for evaluation"

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    results_file = os.path.join(args.results_dir, args.exp_name + "_ssd.txt")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(results_file, "a"))
    logger.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # create model
    if args.training_mode in ["SimCLR", "SupCon"]:
        model = SSLResNet(arch=args.arch).eval()
    elif args.training_mode == "SupCE":
        model = SupResNet(arch=args.arch, num_classes=args.classes).eval()
    else:
        raise ValueError("Provide model class")
    model.encoder = nn.DataParallel(model.encoder).to(device)

    # load checkpoint
    ckpt_dict = torch.load(args.ckpt, map_location="cpu")
    if "model" in ckpt_dict.keys():
        ckpt_dict = ckpt_dict["model"]
    if "state_dict" in ckpt_dict.keys():
        ckpt_dict = ckpt_dict["state_dict"]
    model.load_state_dict(ckpt_dict)

    # dataloaders
    train_loader, test_loader, _ = data.MSCOCO_image(args.data_mode,args.data_dir, user_dict, batch_size=args.batch_size, normalize=args.normalize, size=args.size)

    features_train = get_features(
        model.encoder, train_loader
    )  # using feature befor MLP-head
    features_test = get_features(model.encoder, test_loader)
    print("In-distribution features shape: ", features_train.shape, features_test.shape)

    # ds = ["cifar10", "cifar100", "svhn", "texture", "blobs"]
    # ds.remove(args.dataset)

    # for d in ds:
    ood_loader,_,_ = data.MSCOCO_image(args.data_mode,args.data_dir, user_dict, batch_size=args.batch_size, normalize=args.normalize, size=args.size, F = "OOD")
    features_ood = get_features(model.encoder, ood_loader)
    print("Out-of-distribution features shape: ", features_ood.shape)

    fpr95, auroc, aupr = get_eval_results(
        np.copy(features_train),
        np.copy(features_test),
        np.copy(features_ood),
        args,
    )

    logger.info(
        f"In-data = {args.dataset}, Clusters = {args.clusters}, FPR95 = {fpr95}, AUROC = {auroc}, AUPR = {aupr}"
    )


if __name__ == "__main__":
    main()
