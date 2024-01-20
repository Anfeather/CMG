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





def ssl_with_CMA_CRA(
    textmodel,
    model,
    device,
    dataloader,
    criterion,
    criterion_MSE,
    criterion_CL,
    optimizer,
    optimizer_text,
    optimizer_cross,
    lr_scheduler=None,
    lr_scheduler_cross=None,
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


    end = time.time()

    for i, data in enumerate(dataloader):
        images, caps, _ = data#.cuda()
        

        gmm = GMM(n_components=5).fit(caps)
        center = gmm.means_

        # label_p = gmm.predict_proba(caps)
        labels = gmm.predict(caps)



        images = torch.cat([images[0], images[1]], dim=0).cuda()
        # caps = torch.cat([caps[0], caps[1]], dim=0).cuda()
        caps = caps.cuda()


        bsz = images.shape[0]//2
        image_single = images[:bsz]

        model.train()
        # textmodel.train()
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, torch.from_numpy(labels))

        # losses.update(loss.item(), bsz)
        # print("1",loss)
        # cap_features = caps
        # cap_features = textmodel(caps)
        # f1 = torch.nn.functional.normalize(f1,dim=1)
        # cap_features = torch.nn.functional.normalize(cap_features,dim=1)
        # features_cross = torch.cat([f1.unsqueeze(1), caps.unsqueeze(1)], dim=1)
        # f1 = F.normalize(f1, dim=1)
        # cap_features = F.normalize(cap_features, dim=1)
        # loss =  criterion_MSE(f1, caps) #+ 0.5 * criterion_MSE(f2, caps)
        # features = torch.cat([f1.unsqueeze(1), cap_features.unsqueeze(1)], dim=1)
        # loss_cross = criterion(features_cross, torch.from_numpy(labels))
        loss_cross = criterion_MSE(f1, caps)
        # print("A:",loss)
        # print("B:",loss_cross)
        loss = loss + 0.1 * loss_cross
        optimizer.zero_grad()
        loss.backward()#retain_graph=True
        optimizer.step()
        # if lr_scheduler:
        #     lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)



        # textmodel.train(), model.train()
        # cap_features = textmodel(caps)
        # f1 = model(image_single)
        # # f1 = torch.nn.functional.normalize(f1,dim=1)
        # # cap_features = torch.nn.functional.normalize(cap_features,dim=1)
        # # # features_cross = torch.cat([f1.unsqueeze(1), cap_features.unsqueeze(1)], dim=1)
        # f1 = F.normalize(f1, dim=1)
        # cap_features = F.normalize(cap_features, dim=1)

        # loss_cross = criterion_CL(f1, cap_features)
        # optimizer_cross.zero_grad()
        # loss_cross.backward(retain_graph=True)
        # optimizer_cross.step()
        # cap_features = textmodel(caps)
        # # f1_cap, f2_cap = torch.split(cap_features, [bsz, bsz], dim=0)

        # If1 = torch.nn.functional.normalize(f1.clone().detach(),dim=1)
        # If2 = torch.nn.functional.normalize(f2.clone().detach(),dim=1)
 
        # cap_features = torch.nn.functional.normalize(cap_features,dim=1)
        # # cap_features = torch.cat([f1_cap.unsqueeze(1), f2_cap.unsqueeze(1)], dim=1)

        # loss_cross_1 = criterion_MSE(If1,cap_features)
        # loss_cross_2 = criterion_MSE(If2,cap_features)
        # loss_cross = loss_cross_1 + loss_cross_2
        # optimizer_text.zero_grad()
        # loss_cross.backward(retain_graph=True)
        # optimizer_text.step()
        # if lr_scheduler:
        #     lr_scheduler.step()

        

        ###generate samples ###
        ###generate samples ###
        ###generate samples ###

        model.eval()#,textmodel.eval()
        with torch.no_grad():
            features = model(image_single)
            # cap_features = textmodel(caps)
            cap_features = caps.detach()
        # print(cap_features.shape)
        # print(features.shape)
        # features = torch.nn.functional.normalize(features,dim=1)
        # cap_features = torch.nn.functional.normalize(cap_features,dim=1)
        # f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        # print(features.shape)
        # print(cap_features.shape)
            z_j = F.normalize(features, dim=1)
            z_i = F.normalize(cap_features, dim=1)
        # representations = torch.cat([z_i, z_j], dim=0)
        # cor_matrix = F.cosine_similarity(z_i.unsqueeze(1), z_j.unsqueeze(0), dim=2)
        # cor_matrix = F.cosine_similarity(z_i, z_j, dim=1)
            cor_matrix = torch.abs(torch.mm( z_j, z_i.t() ))
            cor_matrix = cor_matrix - torch.diag( torch.diag(cor_matrix) ) + 10 * torch.eye(cor_matrix.shape[0]).cuda()
            # cor_matrix = torch.mul(mask.cuda(), cor_matrix)

            # cor_matrix = nn.Softmax(cor_matrix,dim=1)
            cor_matrix = cor_matrix / torch.sum(cor_matrix,dim=1)
            # cor_matrix  = F.normalize(cor_matrix, p=1, dim=1)
            # print(cor_matrix)
            generated_batch = []
            for cap_i in cor_matrix:
                generate_img = torch.zeros(image_single[0].shape).cuda()
                for imgid, cap_w in enumerate(cap_i):
                    # if cap_w == 0.0:
                    #     continue

                    generate_img += cap_w * image_single[imgid]
                    
                generated_batch.append(generate_img.detach())
        # print(len(generated_batch))


        # torch.cuda.empty_cache()

        model.train()#,textmodel.eval()
        generated_batch = torch.tensor([item.cpu().detach().numpy() for item in generated_batch]).cuda()
        generated_I_features = model(generated_batch)
        # cap_features = textmodel(caps)
        # generated_I_features = torch.nn.functional.normalize(generated_I_features,dim=1)
        # cap_features = torch.nn.functional.normalize(cap_features,dim=1)
        # f2 = model(images[bsz:])

       
        features = torch.cat([generated_I_features.unsqueeze(1), f2.detach().unsqueeze(1)], dim=1)
        
        # loss = criterion(features)
        loss = criterion(features)
        # print("1",loss)

        # print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler:
            lr_scheduler.step()
            # lr_scheduler_cross.step()






def ssl_with_CMA_FG(
    model,
    device,
    dataloader,
    criterion,
    criterion_MSE,
    criterion_CL,
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


    end = time.time()

    for i, data in enumerate(dataloader):
        images, caps, _ = data#.cuda()

        # gmm = GMM(n_components=5).fit(caps)
        # center = gmm.means_
        # # label_p = gmm.predict_proba(caps)
        # label = gmm.predict(caps)




        images = torch.cat([images[0], images[1]], dim=0).cuda()

        
        bsz = images.shape[0]//2

        model.train()
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        # loss = criterion(features,torch.from_numpy(label))
        loss = criterion(features)
        losses.update(loss.item(), bsz)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if lr_scheduler:
        #     lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


        if lr_scheduler:
            lr_scheduler.step()



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
        
        loss =  criterion_MSE(f1,caps.cuda()) +  criterion_MSE(f2,caps.cuda()) # For Wiki, due to the vast span of the data
        # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        # loss = criterion(features, torch.from_numpy(labels)) # For COCO, UCM


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
        # cro_features = torch.cat([f1.unsqueeze(1), caps.unsqueeze(1).cuda()], dim=1)
        # loss_cross =  criterion_MSE(f1,caps.cuda()) +  criterion_MSE(f2,caps.cuda())
        # loss = loss_cross
        # loss = criterion(features, torch.from_numpy(labels))

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



