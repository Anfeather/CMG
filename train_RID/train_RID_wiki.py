import time
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import classification_ResNet,RankNet
from datasets import *
from utils import *
from tqdm import tqdm
import data_load
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
import sklearn.metrics as skm
import clip
from PIL import Image
import copy
# Data parameters
data_folder = '/home/an/project/data/wikipedia_class/'  # folder with data files saved by create_input_files.py
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
epochs = 100  # number of epochs to train for (if early stopping is not triggered)
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


def validate( predictW, data_loader, criterion):

    batch_l = len(data_loader)
    predictW.eval()
    correct = 0
    test_loss = 0
    l = 0
    for i, (imgs, probs) in tqdm(enumerate(data_loader)):
        l += probs.shape[0]
        # print(len(imgs),imgs[0].shape)

        # Move to GPU, if available
        # img = imgs.to(device)
        
        probs = probs.to(device)
        # print(probs.argmax(1).shape)
        with torch.no_grad():
            pred = predictW(imgs[0].cuda(),imgs[1].cuda(),imgs[2].cuda(),imgs[3].cuda(),imgs[4].cuda(),imgs[5].cuda(),imgs[6].cuda(),imgs[7].cuda(),imgs[8].cuda())
            test_loss += criterion(pred, probs.argmax(1)).item()
            correct += (pred.argmax(1) == probs.argmax(1)).type(torch.float).sum().item()

    print("l:",l)
    return correct/l, test_loss/batch_l 









def main():
    """
    Training and validation.
    """

    # model_clip, preprocess = clip.load("ViT-B/32", device = "cuda")


    predictW = RankNet(10).to(device)
    # predictW = torch.load('/home/an/project/cross-model/ID/pretrain_Wpredictor/wiki/predictWnew.pt', map_location=torch.device('cuda'))
    predictW_optimizer = torch.optim.Adam(predictW.parameters(),lr=encoder_lr)
    # predictW_optimizer = torch.optim.SGD(
    #     predictW.parameters(),
    #     lr=1e-4,
    #     momentum=0.9,
    #     weight_decay=1e-4,
    # )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])



    user_dict = {0:[9,1,3,2],1:[8,4,6,5,0,7]}
 
    train_loader, test_loader, _ = data_load.wiki_image("ssl",data_folder, user_dict, batch_size=batch_size, normalize=True,F="N",FineGrain=1)

    # Epochs
    Loss = []
    Accuracy = []
    acc,loss = validate(predictW, test_loader,criterion)
    Accuracy.append(acc)
    Loss.append(loss)
    print(Accuracy)
    print(Loss)
    best = 0
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        train(train_loader=train_loader,
              criterion=criterion,
              epoch=epoch,
              predictW = predictW,
              predictW_optimizer = predictW_optimizer)
        
        acc,loss = validate(predictW, test_loader,criterion)
        if acc > best:
            print(epoch,best,acc)
            best = acc
            torch.save(predictW,"/home/an/project/cross-model/ID/pretrain_Wpredictor/wiki/predictWnew.pt")
        # else:
        #     predictW = torch.load('/home/an/project/cross-model/ID/pretrain_Wpredictor/wiki/predictWnew.pt', map_location=torch.device('cuda'))        
        Accuracy.append(acc)
        Loss.append(loss)
        print(epoch,max(Accuracy), Accuracy)
        print(Loss)






def train(train_loader, criterion, epoch, predictW, predictW_optimizer):

    predictW.train()
    # Batches
    for i, (imgs, probs) in enumerate(train_loader):

        # print(len(imgs),imgs[0].shape)

        # Move to GPU, if available
        # img = imgs.to(device)
        probs = probs.to(device)

        predicted_data_t = predictW(imgs[0].cuda(),imgs[1].cuda(),imgs[2].cuda(),imgs[3].cuda(),imgs[4].cuda(),imgs[5].cuda(),imgs[6].cuda(),imgs[7].cuda(),imgs[8].cuda())

        # print(predicted_data_t.shape,temp_probs.unsqueeze(1).shape)

        loss_w = criterion(predicted_data_t,probs.argmax(1))
        predictW_optimizer.zero_grad()
        loss_w.backward()
        predictW_optimizer.step()





if __name__ == '__main__':
    main()
