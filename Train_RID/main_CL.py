import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
import sklearn.metrics as skm
from modules import VectorQuantizedVAE, to_scalar, Encoder
from datasets import MiniImagenet
import data_load
from tensorboardX import SummaryWriter
from losses import *
import clip
import json
from tqdm import tqdm



criterion_CL = ContrastiveLoss()
criterion = SupConLoss(temperature=0.5).cuda()
# with open("/home/an/project/data/MS_COCO/image_caption_data_small/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json",'r') as fw:
#     words_map = json.load(fw)

# num2words={}
# for key, value in words_map.items():
#     num2words[value] = key
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

def get_fpr(xin, xood):
    return np.sum(xood < np.percentile(xin, 95)) / len(xood)

def get_scores(ftrain, ftest, food, args):
    # if args.clusters == 1:
        return get_scores_one_cluster(ftrain, ftest, food)
    # else:
    #     if args.training_mode == "SupCE":
    #         print("Using data labels as cluster since model is cross-entropy")
    
    #     else:
    #         ypred = get_clusters(ftrain, args.clusters)
    #     return get_scores_multi_cluster(ftrain, ftest, food, ypred)

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





def train(epoch, data_loader, model,encoder,  optimizer, optimizer_encoder, args):
    for images, captions, _ in data_loader:
        images = images.cuda()
        captions = captions.cuda()


        # sentence_list = []
        # for j, c in enumerate(captions):
        #     sentence = [num2words[int(num)] for num in c if num not in [9489,9488,0]] 
        #     sen = ' '
        #     sen = sen.join(sentence)
        #     sentence_list.append(sen)
        # text = clip.tokenize(sentence_list).to(device)
        # print(text)
        # with torch.no_grad():
        #     # image_features = torch.tensor([model_clip.encode_image(preprocess(toPIL(i)).unsqueeze(0).to(device)).cpu().numpy() for i in imgs])
        #     # image_features = image_features.squeeze(1).to(device)
        #     # image_features = model_clip.encode_image(ori_imgs)
        #     text_features = model_clip.encode_text(text)




        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x = model(captions)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, captions)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
        loss = loss_recons + loss_vq + args.beta * loss_commit #+ loss_recons_image +  args.beta * loss_commit_image 

        loss.backward()
        optimizer.step()




        optimizer_encoder.zero_grad()
        z_e_x_image = encoder(images)

        loss_commit_image = F.mse_loss(z_e_x_image, z_e_x.detach())
        # Commitment objective
        # loss_commit_image = criterion_CL(z_e_x_image, z_e_x.detach())

        # features = torch.cat([z_e_x_image.unsqueeze(1), z_e_x.unsqueeze(1).detach()], dim=1)
        # loss_commit_image = criterion(features) 
        # print(loss_commit_image)
        loss_commit_image.backward()
        optimizer_encoder.step()



        args.steps += 1

def test(data_loader, model, args):
    torch.cuda.empty_cache() 
    loss_recons = []
    with torch.no_grad():
        loss_recon_ = 0.
        for images, _, _ in data_loader:
            images = images.cuda()

            # captions = captions.cuda()
            num_ele = 1.0
            for i in images.shape:
                num_ele *= i
            batch = images.shape[0]
            x_tilde, z_e_x, z_q_x = model(images)
            loss_recon_ += F.mse_loss(x_tilde, images)
            temp_loss = torch.sum(F.mse_loss(x_tilde.reshape(batch,-1), images.reshape(batch,-1),reduce=False),dim=1) / num_ele

            loss_recons.extend(temp_loss.tolist())

            # loss_vq += F.mse_loss(z_q_x, z_e_x)

    #     loss_recons /= len(data_loader)
    #     loss_vq /= len(data_loader)
    return loss_recons, loss_recon_





def get_features(model, data_loader):
    latents = []
    with torch.no_grad():
        loss_recon_ = 0.
        for images, _, _ in data_loader:
            images = images.cuda()

            latents += list(model(images).cpu().numpy())

    return np.array(latents)



def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.cuda()
        x_tilde, _, _ = model(images)
    return x_tilde

def main(args):



    transform = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    user_dict = {0:[ 6, 14, 22, 74, 4, 23], 1:[67, 69, 33, 72, 49, 2, 55, 28, 76, 54, 24, 35, 78, 26, 42, 73, 32, 38, 64, 34, 70, 44, 57, 40, 30, 31, 41, 45, 29, 60, 56, 43, 58, 36, 39, 66, 63, 37, 27, 62, 51, 79, 65, 68, 47, 1, 12, 25, 52, 48]}
    train_loader , _, _ = data_load.Class_COCO("train",args.data_folder, user_dict, batch_size=args.batch_size, size=args.size)
    infer_train_loader, infer_test_loader, _ = data_load.Class_COCO("base",args.data_folder, user_dict, batch_size=args.batch_size, size=args.size)
    ood_loader,_,_ = data_load.Class_COCO("base",args.data_folder, user_dict, batch_size=args.batch_size, size=args.size, F = "OOD")



    encoder = Encoder().cuda()
    model = VectorQuantizedVAE(3, args.hidden_size, args.k).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_encoder = torch.optim.SGD(
        encoder.parameters(),
        lr=1e-4,
        momentum=0.9,
        weight_decay=1e-4,
    )
    # model_clip, preprocess = clip.load("ViT-B/32", device=device)





    # test_losses = []
    best_loss = -1.
    AUROC = []
    AUPR = []
    for epoch in tqdm(range(args.num_epochs)):
        train(epoch, train_loader, model, encoder,  optimizer, optimizer_encoder, args)
        # loss_temp, test_loss_ = test(valid_loader, model, args)
        # test_losses.append(test_loss_.item())


        # xin,_ = test(infer_train_loader, encoder, args)
        # xood,_ = test(ood_loader, encoder, args)
        # # print(xood)
        # norm = get_pred(xin,xood)
        # auroc = get_roc_sklearn(xin, xood)
        # aupr = get_pr_sklearn(xin, xood)


        # if  epoch % 9 == 0: 
        print()
        print(epoch)
        features_train = get_features(
            encoder, infer_train_loader
        )  # using feature befor MLP-head
        features_test = get_features(encoder, infer_test_loader)
        features_ood = get_features(encoder, ood_loader)


        fpr95, auroc, aupr = get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            args,
        )





        AUROC.append(auroc)
        AUPR.append(aupr)
        print("AUROC:",AUROC)
        print("AUPR:",AUPR)
    # loss,_ = test(valid_loader, model, args)
    # print(loss)
    # print(test_losses)
    print("AUROC:",max(AUROC))
    print("AUPR:",max(AUPR))
        # reconstruction = generate_samples(fixed_images, model, args)
        # grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
        # writer.add_image('reconstruction', grid, epoch + 1)
        # if (epoch == 0) or (loss < best_loss):
        #     best_loss = loss
        #     with open('{0}/best.pt'.format(save_filename), 'wb') as f:
        #         torch.save(model.state_dict(), f)
        # with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
        #     torch.save(model.state_dict(), f)

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp


    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    parser.add_argument('--data-folder', type=str, default="/home/an/project/data/class_COCO/anomaly_detection/non_iid_MSCOCO_train_30_50_5images/")
    parser.add_argument('--dataset', type=str)

    # Latent space
    parser.add_argument('--hidden-size', type=int, default=256,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=512,
        help='number of latent vectors (default: 512)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=32,
        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100,
        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-4,
        help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--beta', type=float, default=1.0,
        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='vqvae',
        help='name of the output folder (default: vqvae)')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))

    parser.add_argument("--size", type=int, default=32)
    args = parser.parse_args()

    # Create logs and models folder if they don't exist

    # Device


    args.steps = 0






    main(args)
