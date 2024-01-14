import os
import sys
import numpy as np
import math
import time
import shutil, errno
from distutils.dir_util import copy_tree
import sklearn.metrics as skm
from sklearn.covariance import ledoit_wolf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from PIL import Image
import random

#### logging ####
def save_checkpoint(state, is_best, results_dir, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(results_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(results_dir, filename),
            os.path.join(results_dir, "model_best.pth.tar"),
        )


def create_subdirs(sub_dir):
    os.mkdir(sub_dir)
    os.mkdir(os.path.join(sub_dir, "checkpoint"))


def clone_results_to_latest_subdir(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    copy_tree(src, dst)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


#### evaluation ####
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# def get_features(model, dataloader, max_images=10 ** 10, verbose=False):
#     features = []
#     total = 0
    
#     model.eval()
    
#     for index, img in enumerate(dataloader):

#         if total > max_images:
#             break
#         # print(len(img))
#         # print(img[0].shape)
#         img= torch.tensor([item.cpu().detach().numpy() for item in img]).cuda() 
#         # img = torch.FloatTensor( img ).cuda()

#         features += list(model(img).data.cpu().numpy())
#         # labels += list(label.data.cpu().numpy())

#         if verbose and not index % 50:
#             print(index)

#         total += len(img)

#     return np.array(features)
def get_features_other_arc(model, dataloader, max_images=10 ** 10, verbose=False):
    features = []
    total = 0
    model.eval()
    for index, data in enumerate(dataloader):
        img, caps = data
        if total > max_images:
            break
        # print(len(img))
        # print(img[0].shape)
        img= torch.tensor([item.cpu().detach().numpy() for item in img]).cuda() 
        # img = torch.FloatTensor( img ).cuda()
        # print(img.shape)
        # print(img.squeeze(0).shape)
        features += list(model.encoder(img.squeeze(0)).data.cpu().numpy())
        # labels += list(label.data.cpu().numpy())
        if verbose and not index % 50:
            print(index)

        total += len(img)
    return np.array(features)
    
def get_features(model, dataloader, max_images=10 ** 10, verbose=False):
    features = []
    total = 0
    # model.eval()
    with torch.no_grad():
        for index, data in enumerate(dataloader):
            img, caps,_ = data
            if total > max_images:
                break

            # print(len(img))
            # print(img[0].shape)
            # img= torch.tensor([item.cpu().detach().numpy() for item in img]).cuda() 
            img = torch.tensor(np.array(img)).cuda() 
            # print(img.shape)
            # print(img.squeeze(0).shape)
            # features += list(model(img.squeeze(0)).data.cpu().numpy())
            features += list(model(img).data.cpu().numpy())
            # labels += list(label.data.cpu().numpy())
            if verbose and not index % 50:
                print(index)

            total += len(img)
    return np.array(features)



def get_intermerdiate_features(model, dataloader, max_images=10 ** 10, verbose=False):
    features = []
    total = 0
    model.eval()
    with torch.no_grad():
        for index, data in enumerate(dataloader):
            img, caps,_ = data
            if total > max_images:
                break

            # print(len(img))
            # print(img[0].shape)
            # img= torch.tensor([item.cpu().detach().numpy() for item in img]).cuda() 
            img = torch.tensor(np.array(img)).cuda() 
            # print(img.shape)
            # print(img.squeeze(0).shape)
            # features += list(model(img.squeeze(0)).data.cpu().numpy())
            features += list(model(img).data.cpu().numpy())
            # labels += list(label.data.cpu().numpy())
            if verbose and not index % 50:
                print(index)

            total += len(img)
    return np.array(features)

    
def get_features_clip(dataloader, max_images=10 ** 10, verbose=False):
    features = []
    total = 0

    with torch.no_grad():
        for index, data in enumerate(dataloader):
            imgs, _, _ = data
            if total > max_images:
                break
        # print(imgs.shape,imgs.squeeze().shape)
            features += list(imgs.numpy())
    return np.array(features)



def get_features_FG(model, dataloader, predictW , max_images=10 ** 10, verbose=False):
    features = []
    total = 0
    model.eval()
    with torch.no_grad():
        for index, data in enumerate(dataloader):
            img, _, _ = data
            if total > max_images:
                break
            # print(len(img))
            # print(img[0].shape)


            cut_num = 2
            # 读取要分割的图片，以及其尺寸等数据
            (batch, depth, width, length ) = img.shape
            cut_width = int(width / cut_num)
            cut_length = int(length / cut_num)
            # 预处理生成0矩阵

            temp_img = np.zeros((batch, depth, width, length))

            logits_per_text = np.zeros((1))
            temp_pic = []
            count = 0
            for w in range(0, cut_num):
                for j in range(0, cut_num):
                    pic = np.zeros((batch, depth, cut_width, cut_length ))
                    pic = img[:,:,w*cut_width : (w+1)*cut_width, j*cut_length : (j+1)*cut_length]    
                    predicted_data_t = predictW(pic)
                    # predicted_data_t = np.repeat(predicted_data_t, depth* width*length, axis=1).reshape(batch, depth, width, length )
                    print(pic)
                    print(predicted_data_t )

            img= torch.tensor([item.cpu().detach().numpy() for item in img]).cuda() 
            # img = torch.FloatTensor( img ).cuda()
            # print(img.shape)
            # print(img.squeeze(0).shape)
            # features += list(model(img.squeeze(0)).data.cpu().numpy())
            features += list(model(img).data.cpu().numpy())
            # labels += list(label.data.cpu().numpy())
            if verbose and not index % 50:
                print(index)

            total += len(img)
    return np.array(features)


def get_features_flower_text(model, dataloader, max_images=10 ** 10, verbose=False):
    features = []
    total = 0
    model.eval()
    for index, data in enumerate(dataloader):
        img, caps = data
        if total > max_images:
            break
        # print(len(img))
        # print(img[0].shape)
        caps= torch.tensor([item.cpu().detach().numpy() for item in caps]).cuda() 
        # img = torch.FloatTensor( img ).cuda()
        # print(img.shape)
        # print(img.squeeze(0).shape)
        features += list(model(caps.squeeze(0)).data.cpu().numpy())
        # labels += list(label.data.cpu().numpy())
        if verbose and not index % 50:
            print(index)

        total += len(caps)
    return np.array(features)

def get_features_CMA(model, textmodel, dataloader, max_images=10 ** 10, verbose=False):
    features = []
    total = 0
    model.eval(), textmodel.eval()
    for index, data in enumerate(dataloader):
        img, caps, _ = data
        if total > max_images:
            break
        # print(len(img))
        # print(img[0].shape)
        img = img.cuda()
        caps = caps.cuda()
        # img= torch.tensor([item.cpu().detach().numpy() for item in img]).cuda() 
        cap_features = textmodel(caps)
        # img = torch.FloatTensor( img ).cuda()
        # print(img.shape)
        # print(img.squeeze(0).shape)


        img_features = model(img.squeeze(0)) 
        cap_features = textmodel(caps)
        z_i = F.normalize(img_features, dim=1)
        z_j = F.normalize(cap_features, dim=1)

        cor_matrix = torch.abs(torch.mm( z_j, z_i.t() ))
        cor_matrix = cor_matrix / torch.sum(cor_matrix,dim=1)
        # print(cor_matrix)
        generated_batch = []
        for cap_i in cor_matrix:
            generate_img = torch.zeros(img[0].shape).cuda()
            for imgid, cap_w in enumerate(cap_i):
                generate_img += cap_w * img[imgid]
                
            generated_batch.append(generate_img)
        # print(len(generated_batch))


        generated_batch = torch.tensor([item.cpu().detach().numpy() for item in generated_batch]).cuda() 
        generated_I_features = model.encoder(generated_batch)

        features += list( model.encoder(img).data.cpu().numpy()) + list(generated_I_features.data.cpu().numpy())
        # labels += list(label.data.cpu().numpy())
        if verbose and not index % 50:
            print(index)

        total += len(img) * 2
    return np.array(features)

def get_features_text(model, dataloader, max_images=10 ** 10, verbose=False):
    features = []
    total = 0
    with torch.no_grad():
        for index, data in enumerate(dataloader):
            caps,_ = data
            caps = caps.cuda()
            if total > max_images:
                break
            # print(len(img))
            # print(img[0].shape)
            # img= torch.tensor([item.cpu().detach().numpy() for item in img]).cuda() 
            # img = torch.FloatTensor( img ).cuda()
            # print(img.shape)
            # print(img.squeeze(0).shape)

            features += list(model(caps).data.cpu().numpy())
            # labels += list(label.data.cpu().numpy())
            if verbose and not index % 50:
                print(index)

            total += len(caps)
    return np.array(features)

def baseeval(model, device, val_loader, criterion, args, epoch=0):
    """
    Evaluating on validation set inputs.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

        progress.display(i)  # print final results

    return top1.avg, top5.avg


def knn(model, device, val_loader, criterion, args, writer, epoch=0):
    """
    Evaluating knn accuracy in feature space.
    Calculates only top-1 accuracy (returns 0 for top-5)
    """

    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1]

            # compute output
            output = F.normalize(model(images), dim=-1).data.cpu()
            features.append(output)
            labels.append(target)

        features = torch.cat(features).numpy()
        labels = torch.cat(labels).numpy()

        cls = KNeighborsClassifier(20, metric="cosine").fit(features, labels)
        acc = 100 * np.mean(cross_val_score(cls, features, labels))

        print(f"knn accuracy for test data = {acc}")

    return acc, 0


#### OOD detection ####
def get_roc_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = skm.roc_auc_score(labels, data)
    return auroc


def get_pr_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    aupr = skm.average_precision_score(labels, data)
    return aupr


def get_fpr(xin, xood):
    return np.sum(xood < np.percentile(xin, 95)) / len(xood)


def get_scores_one_cluster(ftrain, ftest, food, shrunkcov=False):
    if shrunkcov:
        print("Using ledoit-wolf covariance estimator.")
        cov = lambda x: ledoit_wolf(x)[0]
    else:
        cov = lambda x: np.cov(x.T, bias=True)

    # ToDO: Simplify these equations
    dtest = np.sum(
        (ftest - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (ftest - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )

    dood = np.sum(
        (food - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (food - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )

    return dtest, dood


#### Dataloaders ####
def readloader(dataloader):
    images = []
    labels = []
    for img, label in dataloader:
        images.append(img)
        labels.append(label)
    return torch.cat(images), torch.cat(labels)


def unnormalize(x, norm_layer):
    m, s = (
        torch.tensor(norm_layer.mean).view(1, 3, 1, 1),
        torch.tensor(norm_layer.std).view(1, 3, 1, 1),
    )
    return x * s + m


class ssdk_dataset(torch.utils.data.Dataset):
    def __init__(self, images, norm_layer, copies=1, s=32):
        self.images = images

        # immitating transformations used at training self-supervised models 
        # replace it if training models with a different data augmentation pipeline
        self.tr = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(s, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                norm_layer,
            ]
        )

        self.n = len(images)
        self.size = len(images) * copies

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.tr(self.images[idx % self.n]), 0


def sliceloader(dataloader, norm_layer, k=1, copies=1, batch_size=128, size=32):

    images, labels = readloader(dataloader)
    indices = np.random.permutation(np.arange(len(images)))
    images, labels = images[indices], labels[indices]

    index_k = torch.cat(
        [torch.where(labels == i)[0][0:k] for i in torch.unique(labels)]
    ).numpy()
    index_not_k = np.setdiff1d(np.arange(len(images)), index_k)

    dataset_k = ssdk_dataset(
        unnormalize(images[index_k], norm_layer), norm_layer, copies, size
    )
    dataset_not_k = torch.utils.data.TensorDataset(
        images[index_not_k], labels[index_not_k]
    )
    print(
        f"Number of selected OOD images (k * num_classes_ood_dataset) = {len(index_k)} \nNumber of OOD images after augmentation  = {len(dataset_k)} \nRemaining number of test images in OOD dataset = {len(dataset_not_k)}"
    )

    loader_k = torch.utils.data.DataLoader(
        dataset_k, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    loader_not_k = torch.utils.data.DataLoader(
        dataset_not_k,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return loader_k, loader_not_k


def data_assign(num_users):
    data_distribution = list(range(80))
    assign = {}
    for i in range(num_users):
        assign[i] = []

    for i in range(num_users):
        temp = random.sample(data_distribution, 80//num_users)
        for j in temp:
            data_distribution.remove(j)
        assign[i].extend(temp)
    return assign


def get_near(X):
    l,h = X.shape




def randomNumVaccinePersonTotal(maxValue, num):
    '''生成总和固定的整数序列
    maxValue: 序列总和
    num：要生成的整数个数

    return
    per_all_persons:list,指定 num个接种点各自待接种的人数
    '''
    alpha = 1.0
    lam = np.random.beta(alpha, alpha)
    print(lam)
    sample_list = np.random.uniform(0,lam,(1,10))
    print(sample_list)

    maxValue = int(maxValue)
    suiji_ser = random.sample(range(1,maxValue), k=num-1)  # 在1~maxValue之间，采集20个数据
    suiji_ser.append(0)   # 加上数据开头
    suiji_ser.append(maxValue)
    suiji_ser = sorted(suiji_ser)
    suiji_ser = np.array(suiji_ser) / 1000
    per_all_persons = [ suiji_ser[i]-suiji_ser[i-1] for i in range(1, len(suiji_ser)) ] # 列表推导式，计算列表中每两个数之间的间隔

    return per_all_persons