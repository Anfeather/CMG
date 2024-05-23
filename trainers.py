import torch
import torch.nn as nn
import time
from utils import AverageMeter, ProgressMeter, accuracy
import torch.nn.functional as F
from torch import nn
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from torch.autograd import Variable

import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)





def ssl_CMLE(
    model,
    device,
    dataloader,
    criterion,
    criterion_CL,
    criterion_MSE,
    optimizer,
    lr_scheduler=None,
    epoch=0,
    args=None,
):
    print(
        " ->->->->->->->->->-> One epoch with self-supervised training <-<-<-<-<-<-<-<-<-<-"
    )

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()

    for i, data in enumerate(dataloader):
        images,caps,_ = data#.cuda()

        gmm = GMM(n_components=5).fit(caps)
        # center = gmm.means_
        # label_p = gmm.predict_proba(caps)
        labels = gmm.predict(caps)

        images = torch.cat([images[0], images[1]], dim=0).cuda()
        bsz = images.shape[0]//2
        # basic properties of training
        if i == 0:
            print(
                images.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(
                "Pixel range for training images : [{}, {}]".format(
                    torch.min(images).data.cpu().numpy(),
                    torch.max(images).data.cpu().numpy(),
                )
            )

        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        
        #loss =  criterion_MSE(f1,caps.cuda()) +  criterion_MSE(f2,caps.cuda()) 
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, torch.from_numpy(labels)) # For COCO, UCM


        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if lr_scheduler:
            lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # torch.cuda.empty_cache()

        if i % args.print_freq == 0:
            progress.display(i)



def ssl_base(
    model,
    device,
    dataloader,
    criterion,
    criterion_CL,
    criterion_MSE,
    optimizer,
    lr_scheduler=None,
    epoch=0,
    args=None,
):
    print(
        " ->->->->->->->->->-> One epoch with self-supervised training <-<-<-<-<-<-<-<-<-<-"
    )

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()

    for i, data in enumerate(dataloader):
        images,caps,_ = data#.cuda()

        images = torch.cat([images[0], images[1]], dim=0).cuda()
        bsz = images.shape[0]//2
        # basic properties of training
        if i == 0:
            print(
                images.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(
                "Pixel range for training images : [{}, {}]".format(
                    torch.min(images).data.cpu().numpy(),
                    torch.max(images).data.cpu().numpy(),
                )
            )



        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = criterion(features)
   

        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if lr_scheduler:
            lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # torch.cuda.empty_cache()

        if i % args.print_freq == 0:
            progress.display(i)



