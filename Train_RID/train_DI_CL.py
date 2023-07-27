import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention, PredictWNet
from datasets import *
from utils import *
import data_load
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
import sklearn.metrics as skm
from losses import *

criterion_CL = ContrastiveLoss()
criterion_ = SupConLoss(temperature=0.5).cuda()

# Data parameters
data_folder = '/home/an/project/data/MS_COCO/image_caption_data_small'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 128
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = True  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none

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

def get_scores(ftrain, ftest, food):
    # if args.clusters == 1:
        return get_scores_one_cluster(ftrain, ftest, food)
    # else:
    #     if args.training_mode == "SupCE":
    #         print("Using data labels as cluster since model is cross-entropy")
    
    #     else:
    #         ypred = get_clusters(ftrain, args.clusters)
    #     return get_scores_multi_cluster(ftrain, ftest, food, ypred)

def get_eval_results(ftrain, ftest, food):
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

    dtest, dood = get_scores(ftrain, ftest, food)

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

def get_features(encoder,predictW, data_loader):
    latents = []
    with torch.no_grad():
        loss_recon_ = 0.
        for images, _, _ in data_loader:

            imgs = images.cuda()
            

            encoder_out = encoder(imgs)
            batch_size = encoder_out.size(0)
            encoder_dim = encoder_out.size(-1)

            # Flatten image
            encoder_out = encoder_out.reshape(batch_size, -1, encoder_dim) 


            distilled_information = torch.zeros(batch_size,encoder_dim)
            
            

            for i, data_w in enumerate(encoder_out):
                predicted_data_t = predictW(data_w)
                weight_cout = torch.zeros(encoder_dim).cuda()
                for a, b in zip(predicted_data_t,data_w):
                    
                    weight_cout += a * b
                # print(weight_cout.shape)
                # data_w = data_w * predicted_data_t.unsqueeze(2).sum(1)
                # print(weight_cout.shape)
                distilled_information[i] = weight_cout

            



            latents += list(distilled_information.cpu().numpy())

    return np.array(latents)


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)


    data_folder_ = "/home/an/project/data/class_COCO/anomaly_detection/non_iid_MSCOCO_train_30_50_numbercaption/"


    predictW = PredictWNet().cuda()
    predictW_optimizer = torch.optim.Adam(predictW.parameters(),lr=encoder_lr)
    # Move to GPU, if available
    decoder = decoder.cuda()
    encoder = encoder.cuda()
    
    # Loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])




    user_dict = {0:[ 6, 14, 22, 74, 4, 23], 1:[67, 69, 33, 72, 49, 2, 55, 28, 76, 54, 24, 35, 78, 26, 42, 73, 32, 38, 64, 34, 70, 44, 57, 40, 30, 31, 41, 45, 29, 60, 56, 43, 58, 36, 39, 66, 63, 37, 27, 62, 51, 79, 65, 68, 47, 1, 12, 25, 52, 48]}
    train_loader , val_loader, _ = data_load.Class_COCO("train",data_folder_, user_dict, batch_size=batch_size)
    infer_train_loader, infer_test_loader, _ = data_load.Class_COCO("base",data_folder_, user_dict, batch_size=batch_size)
    ood_loader,_,_ = data_load.Class_COCO("base",data_folder_, user_dict, batch_size=batch_size, F = "OOD")

    # Epochs
    AUROC = []
    AUPR = []
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch,
              predictW = predictW,
              predictW_optimizer = predictW_optimizer)

        # # One epoch's validation
        # recent_bleu4 = validate(val_loader=val_loader,
        #                         encoder=encoder,
        #                         decoder=decoder,
        #                         criterion=criterion)




        features_train = get_features(
            encoder, predictW, infer_train_loader
        )  # using feature befor MLP-head
        features_test = get_features(encoder, predictW, infer_test_loader)
        features_ood = get_features(encoder, predictW, ood_loader)


        fpr95, auroc, aupr = get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
        )
        AUROC.append(auroc)
        AUPR.append(aupr)
        print(AUROC)
        print(AUPR)
        # Check if there was an improvement
        # is_best = recent_bleu4 > best_bleu4
        # best_bleu4 = max(recent_bleu4, best_bleu4)
        # if not is_best:
        #     epochs_since_improvement += 1
        #     print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        # else:
        #     epochs_since_improvement = 0

        # Save checkpoint
        # save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
        #                 decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, predictW, predictW_optimizer):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        # imgs = imgs.cuda()
        imgs[0] = imgs[0].cuda()
        imgs[1] = imgs[1].cuda()
        images = torch.cat([imgs[0], imgs[1]], dim=0)

        caps = caps.cuda()
        caplens = caplens.cuda()

        # Forward prop.
        imgs = encoder(imgs[0])
        scores, caps_sorted, decode_lengths, alphas, sort_ind, distilled_information = decoder(imgs, caps, caplens)

        # for i, data_w in enumerate(encoder_out):
        #     predicted_data_t = predictW(data_w)
        #     weight_cout = torch.zeros(encoder_dim).cuda()
        #     for a, b in zip(predicted_data_t,data_w):
                
        #         weight_cout += a * b
        #     # print(weight_cout.shape)
        #     # data_w = data_w * predicted_data_t.unsqueeze(2).sum(1)
        #     # print(weight_cout.shape)
        #     distilled_information[i] = weight_cout



        features = encoder(images)
        batch_size = imgs.size(0)
        encoder_dim = imgs.size(-1)
        # Flatten image

        f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
        f2 = f2.view(batch_size, -1, encoder_dim) 
        f1 = imgs.view(batch_size, -1, encoder_dim) 
        # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        # distilled_information = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        other = alphas.shape[1]
        target_weights = alphas.sum(dim=1)/other
        distilled_information = torch.zeros(batch_size,encoder_dim).cuda()
        distilled_information2 = torch.zeros(batch_size,encoder_dim).cuda()        
        k_t = 0
        for data_w, data_w2, data_t in zip(f1, f2, target_weights):
            predictW_optimizer.zero_grad()
            predicted_data_t = predictW(data_w.detach())

            loss_w = F.mse_loss(data_t.detach(),predicted_data_t)
            loss_w.backward()
            predictW_optimizer.step()

            ID_data = (data_w.detach()*(data_t.unsqueeze(1).repeat(1,2048))).sum(0)
            ID_data2 = (data_w2.detach()*(data_t.unsqueeze(1).repeat(1,2048))).sum(0)
            distilled_information[k_t] = ID_data
            distilled_information2[k_t] = ID_data
            k_t += 1
        features = torch.cat([distilled_information.unsqueeze(1), distilled_information.unsqueeze(1)], dim=1)
        loss_CL = criterion_(features)
        # encoder_optimizer.zero_grad()
        # loss_CL.backward()
        # encoder_optimizer.step()
        # print(loss_CL)







        # print(distilled_information.shape)
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean() + loss_CL

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)







        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.cuda()
            caps = caps.cuda()
            caplens = caplens.cuda()

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind, distilled_information = decoder(imgs, caps, caplens)
            
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
