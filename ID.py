# Some part borrowed from official tutorial https://github.com/pytorch/examples/blob/master/imagenet/main.py
from __future__ import print_function
from __future__ import absolute_import
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import numpy as np
import argparse
import importlib
import time
import logging
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
# from torch.utils.tensorboard import SummaryWriter

from models import  SSLResNet,classification_ResNet
import data
import trainers
from losses import SupConLoss, ContrastiveLoss
from utils import *
import torch.nn.functional as F
import random
# seed = 999
# random.seed(seed)
import numpy as np
# np.random.seed(seed)



import os
import numpy as np
import time
import logging
import argparse
from collections import OrderedDict
import faiss

import torch
import torch.nn as nn

from models import  SSLResNet
from utils import (
    get_features_FG,
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



beta1 = 0.5
beta2 = 0.9
def main():
    parser = argparse.ArgumentParser(description="SSD evaluation")

    parser.add_argument(
        "--results-dir",
        type=str,
        default="/home/ray/preject/cross_model/CMDA/resulrts/trained_models/",
    )  # change this
    parser.add_argument("--exp-name", type=str, default="temp")
    parser.add_argument(
        "--training-mode", type=str, default="SimCLR"
    )

    # model
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--num-classes", type=int, default=10)

    # training
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data-dir", type=str, default="/home/ray/preject/data/class_coco_clip_processed/non_iid_MSCOCO_train_30_50_5images/")
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", action="store_true")

    # ssl
    parser.add_argument(
        "--method", type=str, default="SimCLR", choices=["SupCon", "SimCLR", "SupCE"]
    )
    parser.add_argument("--temperature", type=float, default=0.5)

    # misc
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=50)
    parser.add_argument("--ckpt", type=str, help="checkpoint path")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--clusters", type=int, default=1)
    args = parser.parse_args()
    device = "cuda:0"

    if args.batch_size > 256 and not args.warmup:
        warnings.warn("Use warmup training for larger batch-sizes > 256")


    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)

    user_dict = {0:[6, 14, 22, 74, 4, 23],1:[67, 69, 33, 72, 49, 2, 55, 28, 76, 54, 24, 35, 78, 26, 42, 73, 32, 38, 64, 34, 70, 44, 57, 40, 30, 31, 41, 45, 29, 60, 56, 43, 58, 36, 39, 66, 63, 37, 27, 62, 51, 79, 65, 68, 47, 1, 12, 25, 52, 48]}

    # seed cuda
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # np.random.seed(args.seed)

    # Create model
    model = SSLResNet(arch=args.arch).to(device)

    model.encoder = torch.nn.DataParallel(model.encoder).to(device)

    train_loader, test_loader, _ = data.MSCOCO_image_with_CMA_D("ssl",args.data_dir, user_dict, batch_size=args.batch_size, normalize=args.normalize, size=args.size,F="N",FineGrain=1)
    infer_train_loader, infer_test_loader, _ = data.MSCOCO_image_with_CMA_D("base",args.data_dir, user_dict, batch_size=args.batch_size, normalize=args.normalize, size=args.size)
    ood_loader,_,_ = data.MSCOCO_image_with_CMA_D("base",args.data_dir, user_dict, batch_size=args.batch_size, normalize=args.normalize, size=args.size, F = "OOD")

    criterion = SupConLoss(temperature=args.temperature).cuda()

    criterion_MSE = F.mse_loss
    criterion_CL = ContrastiveLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


    # select training and validation methods
    trainer = (trainers.ssl_with_CMA_FG)
    val = knn #if args.training_mode in ["SimCLR", "SupCon"] else baseeval


    # pretrain_W = torch.load('/home/ray/preject/cross_model/ID/predictW.pt', map_location=torch.device('cpu'))

    best_prec1 = 0

    for p in optimizer.param_groups:
        p["lr"] = args.lr
        p["initial_lr"] = args.lr
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs * len(train_loader), 1e-4
    )



    AUROC_list = []
    AUPR_list = []
    for epoch in range(0, args.epochs):

        trainer(
            model, device, train_loader, criterion,criterion_MSE,criterion_CL, optimizer, lr_scheduler, epoch, args
        )

        # prec1, _ = val(model, device, test_loader, criterion, args, epoch)

        # remember best accuracy and save checkpoint
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)

        d = {
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            # "best_prec1": best_prec1,
            "optimizer": optimizer.state_dict(),
        }



        # test
        
        if epoch%1==0:
  
            features_train = get_features(
                model.encoder, infer_train_loader
            )  
            features_test = get_features(model.encoder, infer_test_loader)
            print("In-distribution features shape: ", features_train.shape, features_test.shape)

            # ds = ["cifar10", "cifar100", "svhn", "texture", "blobs"]
            # ds.remove(args.dataset)

            # for d in ds:
            
            features_ood = get_features(model.encoder, ood_loader)
            print("Out-of-distribution features shape: ", features_ood.shape)

            fpr95, auroc, aupr = get_eval_results(
                np.copy(features_train),
                np.copy(features_test),
                np.copy(features_ood),
                args,
            )


            AUROC_list.append(auroc)
            AUPR_list.append(aupr)
            print("MAX_AUROC:",max(AUROC_list),AUROC_list)
            print("MAX_PR:",max(AUPR_list),AUPR_list)

    

if __name__ == "__main__":
    main()
