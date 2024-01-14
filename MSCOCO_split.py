import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np
from PIL import Image
import copy
import cv2

import torch
import clip
from PIL import Image
from skimage import io
from collections import Counter
device =  "cpu"
model_clip, preprocess = clip.load("ViT-B/32", device=device)






class non_iid_MSCOCODataset_Distillation(Dataset):

    def __init__(self, data_folder, data_name, flag, user_dict, user_id, transform, FineGrain):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        data_distribution = user_dict[user_id]
        # Captions per image
        self.FineGrain = FineGrain
        self.cpi = 1
        self.imgs = []
        self.probs = []
        self.caplens = []
        self.captions = []
        self.captions_words = []
        if flag == "train":
            for i in data_distribution:
                data_train_path = data_folder + str(i)

                # Open hdf5 file where images are stored
                if self.FineGrain == 1:
                    self.h = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name + 'clip_processed_1to1.hdf5'), 'r')
                    self.imgs.extend(self.h['images'])
                    self.probs.extend(self.h['prob'])
                else:
                    self.h = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name + 'clip_processed_1to1.hdf5'), 'r')
                    self.imgs.extend(self.h['images'])                    
                # Load encoded captions (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPTIONS_' + data_name + "vector" + '.json'), 'r') as j:
                    self.captions.extend(json.load(j))
     


                # with open(os.path.join(data_train_path,  '_CAPTIONS_' + data_name  + '.json'), 'r') as j:
                #     self.captions_words.extend(json.load(j))

                # Load caption lengths (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPLENS_' + data_name + '.json'), 'r') as j:
                    self.caplens.extend(json.load(j))
        elif flag == "test":
            for i in data_distribution:
                data_train_path = data_folder + str(i)

                # Open hdf5 file where images are stored
                self.h = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name + 'clip_processed_1to1_test.hdf5'), 'r')
                self.imgs.extend(self.h['images'])

                # Load encoded captions (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPTIONS_' + data_name + "_test" + "vector" + '.json'), 'r') as j:
                    self.captions.extend(json.load(j))

                # with open(os.path.join(data_train_path,  '_CAPTIONS_' + data_name + "_test"  + '.json'), 'r') as j:
                #     self.captions_words.extend(json.load(j))

                # Load caption lengths (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPLENS_' + data_name + "_test" + '.json'), 'r') as j:
                    self.caplens.extend(json.load(j))
            
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

        # print("client data:", self.dataset_size)

    def __getitem__(self, i):
        if self.FineGrain == 1:
            img = self.imgs[i]

            # img = Image.fromarray(img.transpose(1,2,0))
            img = Image.fromarray(np.uint8(img.transpose(1,2,0)))
            img = self.transform(img)

            probs = torch.FloatTensor([])

        else: 
            img = self.imgs[i//5]
            img = Image.fromarray(np.uint8(img.transpose(1,2,0)))
            img = self.transform(img)
            probs = []
        # print(len(self.captions[i]))
        caption =  torch.FloatTensor(self.captions[i])

        caplen = torch.FloatTensor([self.caplens[i]])

        
        return img, caption, probs


    def __len__(self):
        return self.dataset_size





class non_iid_MSCOCODataset(Dataset):

    def __init__(self, data_folder, data_name, flag, user_dict, user_id,transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        data_distribution = user_dict[user_id]
        # Captions per image
        self.cpi = 5
        self.imgs = []
        self.caplens = []
        self.captions = []
        if flag == "train":
            for i in data_distribution:
                data_train_path = data_folder + str(i)

                # Open hdf5 file where images are stored
                self.h = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name + '.hdf5'), 'r')
                self.imgs.extend(self.h['images'])

                # Load encoded captions (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPTIONS_' + data_name + "vector" + '.json'), 'r') as j:
                    self.captions.extend(json.load(j))

                # Load caption lengths (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPLENS_' + data_name + '.json'), 'r') as j:
                    self.caplens.extend(json.load(j))
        elif flag == "test":
            for i in data_distribution:
                data_train_path = data_folder + str(i)

                # Open hdf5 file where images are stored
                self.h = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name+ "_test" + '.hdf5'), 'r')
                self.imgs.extend(self.h['images'])

                # Load encoded captions (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPTIONS_' + data_name + "_test" + "vector" + '.json'), 'r') as j:
                    self.captions.extend(json.load(j))

                # Load caption lengths (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPLENS_' + data_name + "_test" + '.json'), 'r') as j:
                    self.caplens.extend(json.load(j))
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

        # print("client data:", self.dataset_size)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = self.imgs[i//self.cpi] 
        # print("1", img.shape)


        # print(type(img))
        img = Image.fromarray(np.uint8(img.transpose(1,2,0)))
        img = self.transform(img)
    
        # print(len(self.captions[i]))
        caption =  torch.FloatTensor(self.captions[i])

        caplen = torch.FloatTensor([self.caplens[i]])


        return img, caption, caplen


    def __len__(self):
        return self.dataset_size


class non_iid_MSCOCODataset_clip(Dataset):

    def __init__(self, data_folder, data_name, flag, user_dict, user_id,transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        data_distribution = user_dict[user_id]
        # Captions per image
        self.cpi = 5
        self.imgs = []
        self.caplens = []
        self.captions = []
        if flag == "train":
            for i in data_distribution:
                data_train_path = data_folder + str(i)

                # Open hdf5 file where images are stored
                self.h = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name + '.hdf5'), 'r')
                self.imgs.extend(self.h['images'])

                # Load encoded captions (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPTIONS_' + data_name + "vector" + '.json'), 'r') as j:
                    self.captions.extend(json.load(j))

                # Load caption lengths (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPLENS_' + data_name + '.json'), 'r') as j:
                    self.caplens.extend(json.load(j))
        elif flag == "test":
            for i in data_distribution:
                data_train_path = data_folder + str(i)

                # Open hdf5 file where images are stored
                self.h = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name+ "_test" + '.hdf5'), 'r')
                self.imgs.extend(self.h['images'])

                # Load encoded captions (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPTIONS_' + data_name + "_test" + "vector" + '.json'), 'r') as j:
                    self.captions.extend(json.load(j))

                # Load caption lengths (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPLENS_' + data_name + "_test" + '.json'), 'r') as j:
                    self.caplens.extend(json.load(j))
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

        # print("client data:", self.dataset_size)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = self.imgs[i//self.cpi] 

        img = Image.fromarray(np.uint8(img.transpose(1,2,0)))
        image = preprocess(img).unsqueeze(0)
        img_feature = model_clip.encode_image(image)

        # print(len(self.captions[i]))
        caption =  torch.FloatTensor(self.captions[i])

        caplen = torch.FloatTensor([self.caplens[i]])


        return img_feature.squeeze(), caption, caplen


    def __len__(self):
        return self.dataset_size



class TwoCropcaption:
    """Create two crops of the same image"""

    def __init__(self):
        self.transform = None

    def __call__(self, x):
        # print("2", self.transform(x).shape)
        return [x, x]





class Flower_detection(Dataset):

    def __init__(self, data_folder, flag, user_dict, user_id,transform=None,F="N", LG = 0):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        data_distribution = user_dict[user_id]
        # Captions per image
        self.cpi = 10
        self.imgs = []
        self.captions = []
        self.labels = []
        self.F = F
        self.flag = flag

        if F == "N":
            if flag == "train":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image" + '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))

                    with open(os.path.join(data_train_path, "text_struction" + '.json'), 'r') as j:
                        self.labels.extend(json.load(j))

            elif flag == "test":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    with open(os.path.join(data_train_path,  "image_test"+ '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        elif F == "OOD":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image"  + '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))


                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image_test"+ '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.imgs)

        # print("client data:", self.dataset_size)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = "/home/ray/preject/data/Flower_102/jpg/" + self.imgs[i] + "jpg"

        # print("1", img.shape)
        if self.transform is not None:
            # print(type(img))
            # img = Image.fromarray(np.uint8(img.transpose(1,2,0)))
            # img_raw = scipy.misc.imread( os.path.join(img_dir, name) )
            # img = tl.prepro.imresize(img_raw, size=[64, 64])    # (64, 64, 3)
            # img = img.astype(np.float32)
            img = cv2.imread(img)
            img = Image.fromarray(np.uint8(img))
            # img = self.transform_(img)
            
            # print(img.shape)
            img = self.transform(img)
        # print(len(self.captions[i]))
        # caption = torch.sum( torch.FloatTensor(self.captions[i]), dim = 0 ) / 10 
        caption = torch.FloatTensor(self.captions[i*10])
        if self.F == "N" and self.flag == "train":
            labels_item = self.labels[i*10:i*10+10]
            label_fre = Counter(labels_item)

            label = max(label_fre, key = label_fre.get)
            # label = torch.tensor(self.labels[i*10])
        else:
            label = 1
        return img, caption, label


    def __len__(self):
        return self.dataset_size







class Flower_detection_LG(Dataset):

    def __init__(self, data_folder, flag, user_dict, user_id,transform=None,F="N",LG=0):

        data_distribution = user_dict[user_id]
        self.imgs = []
        self.captions = []
        self.LG = LG
        self.F = F
        if F == "N":
            if flag == "train":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    self.h = h5py.File(os.path.join(data_train_path,  "image_processed" + '.hdf5'), 'r')
                    self.imgs.extend(self.h['images'])

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
            elif flag == "test":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    self.h = h5py.File(os.path.join(data_train_path,  "image_processed_test" + '.hdf5'), 'r')
                    self.imgs.extend(self.h['images'])

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        elif F == "OOD":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    self.h = h5py.File(os.path.join(data_train_path,  "image_processed" + '.hdf5'), 'r')
                    self.imgs.extend(self.h['images'])

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))


                    # Load encoded captions (completely into memory)
                    self.h = h5py.File(os.path.join(data_train_path,  "image_processed_test" + '.hdf5'), 'r')
                    self.imgs.extend(self.h['images'])

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        self.dataset_size = len(self.imgs)
 
        # Total number of datapoints
 
        # if self.LG == 1:
        #     self.captions = np.array(self.captions).reshape(-1,1024)
        #     self.dataset_size = self.captions.shape[0]
        # if self.LG == 0 and self.F == "N":
        #     self.dataset_size = len(self.imgs)//10
        # if self.LG == 0 and self.F == "OOD": 
        #     self.dataset_size = len(self.imgs)


    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        # if self.LG == 1:
        #     img = self.imgs[i]
        #     img = Image.fromarray(np.uint8(img.transpose(1,2,0)))
        #     img = self.transform(img)
        #     caption =  torch.FloatTensor(self.captions[i])
        # if self.LG == 0 and self.F == "N":
        #     img = self.imgs[i*10]
        #     img = Image.fromarray(np.uint8(img.transpose(1,2,0)))
        #     img = self.transform(img) 
        #     caption =  []
        # if self.LG == 0 and self.F == "OOD":        
        img = self.imgs[i]
        img = Image.fromarray(np.uint8(img.transpose(1,2,0)))
        img = self.transform(img)
        
        caption = torch.sum( torch.FloatTensor(self.captions[i]), dim = 0 ) / 10 

        label = 1
        return img, caption, label


    def __len__(self):
        return self.dataset_size





class UCM_caption_read(Dataset):

    def __init__(self, data_folder, flag, user_dict, user_id,transform=None,F="N",LG=0):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        data_distribution = user_dict[user_id]
        # Captions per image
        self.cpi = 5
        self.imgs = []
        self.captions = []
        self.LG = LG
        if F == "N":
            if flag == "train":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image" + '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "caption_vector" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
            elif flag == "test":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    with open(os.path.join(data_train_path,  "image_test"+ '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "caption_vector_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        elif F == "OOD":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image"  + '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "caption_vector" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))


                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image_test"+ '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "caption_vector_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform


        # self.captions = np.array(self.captions)


        self.dataset_size = len(self.captions)
        # print(len(self.imgs),len(self.captions))
        # print("client data:", self.dataset_size)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        path_img = "/home/ray/preject/data/UCM_class/imgs/"
        img = path_img + str(self.imgs[i]) + ".tif"
        img = io.imread(img)
        img = Image.fromarray(np.uint8(img))
        img = self.transform(img)

        caption =  torch.FloatTensor(self.captions[i])
        label = 1
        return img, caption, label


    def __len__(self):
        return self.dataset_size




class UCM_caption_read_clip(Dataset):

    def __init__(self, data_folder, flag, user_dict, user_id,transform=None,F="N",LG=0):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        data_distribution = user_dict[user_id]
        # Captions per image
        self.cpi = 5
        self.imgs = []
        self.captions = []
        self.LG = LG
        if F == "N":
            if flag == "train":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image" + '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "caption_vector" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
            elif flag == "test":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    with open(os.path.join(data_train_path,  "image_test"+ '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "caption_vector_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        elif F == "OOD":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image"  + '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "caption_vector" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))


                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image_test"+ '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "caption_vector_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform


        # self.captions = np.array(self.captions)


        self.dataset_size = len(self.captions)
        # print(len(self.imgs),len(self.captions))
        # print("client data:", self.dataset_size)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        path_img = "/home/ray/preject/data/UCM_class/imgs/"
        img = path_img + str(self.imgs[i]) + ".tif"
        img = io.imread(img)
        img = Image.fromarray(np.uint8(img))
        image = preprocess(img).unsqueeze(0)
        img_feature = model_clip.encode_image(image)

        caption =  torch.FloatTensor(self.captions[i])
        label = 1
        return img_feature.squeeze(), caption, label


    def __len__(self):
        return self.dataset_size





class UCM_caption_detection_I2T(Dataset):

    def __init__(self, data_folder, flag, user_dict, user_id,transform=None,F="N",LG=0):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        data_distribution = user_dict[user_id]
        # Captions per image
        self.cpi = 5
        self.imgs = []
        self.captions = []
        self.LG = LG
        if F == "N":
            if flag == "train":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image" + '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "caption_vector" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
            elif flag == "test":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    with open(os.path.join(data_train_path,  "image_test"+ '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "caption_vector_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        elif F == "OOD":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image"  + '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "caption_vector" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))


                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image_test"+ '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "caption_vector_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform


        # self.captions = np.array(self.captions)


        self.dataset_size = len(self.captions)

        # print("client data:", self.dataset_size)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image

        img = "/home/ray/preject/data/UCM_class/imgs/" + self.imgs[i] + ".tif"
        img = io.imread(img)
        img = Image.fromarray(np.uint8(img))
        img = self.transform(img)
      
        caption =  torch.FloatTensor(self.captions[i])

        return img, caption


    def __len__(self):
        return self.dataset_size







class UCM_caption_LG(Dataset):

    def __init__(self, data_folder, flag, user_dict, user_id,transform=None,F="N",LG=0):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        data_distribution = user_dict[user_id]
        # Captions per image
        self.cpi = 5
        self.imgs = []
        self.captions = []
        self.LG = LG
        if F == "N":
            if flag == "train":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    self.h = h5py.File(os.path.join(data_train_path,  'image_processed' + '.hdf5'), 'r')
                    self.imgs.extend(self.h['images'])

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "caption_vector" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
            elif flag == "test":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    self.h = h5py.File(os.path.join(data_train_path,  'image_processed_test' + '.hdf5'), 'r')
                    self.imgs.extend(self.h['images'])

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "caption_vector_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        elif F == "OOD":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    self.h = h5py.File(os.path.join(data_train_path,  'image_processed' + '.hdf5'), 'r')
                    self.imgs.extend(self.h['images'])

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "caption_vector" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))


                    # Load encoded captions (completely into memory)
                    self.h = h5py.File(os.path.join(data_train_path,  'image_processed_test' + '.hdf5'), 'r')
                    self.imgs.extend(self.h['images'])

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "caption_vector_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform


        # self.captions = np.array(self.captions)


        self.dataset_size = len(self.captions)

        # print("client data:", self.dataset_size)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image

        # img = self.imgs[i] + ".tif"
        # img = io.imread(img)
        # print(len(self.imgs),len(self.captions))

        img = Image.fromarray(np.uint8(self.imgs[i].transpose(1,2,0)))
        
        img = self.transform(img)
      
        caption =  torch.FloatTensor(self.captions[i])

        return img, caption, 1


    def __len__(self):
        return self.dataset_size


class wikipedia_read(Dataset):

    def __init__(self, data_folder, flag, user_dict, user_id,transform=None,F="N", LG = 0):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        data_distribution = user_dict[user_id]
        # Captions per image
        self.imgs = []
        self.captions = []
        self.labels = []
        self.F = F
        self.flag = flag

        if F == "N":
            if flag == "train":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image" + '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))

                    with open(os.path.join(data_train_path, "label" + '.json'), 'r') as j:
                        self.labels.extend(json.load(j))

            elif flag == "test":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    with open(os.path.join(data_train_path,  "image_test"+ '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        elif F == "OOD":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image"  + '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))


                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image_test"+ '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.imgs)

        # print("client data:", self.dataset_size)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img =  "/home/ray/preject/data/wikipedia_class/sum_image/"  + self.imgs[i] + ".jpg"

        # print("1", img.shape)
        if self.transform is not None:

            img = cv2.imread(img)
            img = Image.fromarray(np.uint8(img))

            img = self.transform(img)

        caption = torch.FloatTensor(self.captions[i])

        if self.F == "N" and self.flag == "train":
            label = self.labels[i]
        else:
            label = 1
        return img, caption, label


    def __len__(self):
        return self.dataset_size








class wikipedia_LG(Dataset):

    def __init__(self, data_folder, flag, user_dict, user_id,transform=None,F="N", LG = 0):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        data_distribution = user_dict[user_id]
        # Captions per image
        self.imgs = []
        self.captions = []
        self.labels = []
        self.F = F
        self.flag = flag

        if F == "N":
            if flag == "train":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    # with open(os.path.join(data_train_path,  "image" + '.json'), 'r') as j:
                    #     self.imgs.extend(json.load(j))
                    self.h = h5py.File(os.path.join(data_train_path,  'image_processed' + '.hdf5'), 'r')
                    self.imgs.extend(self.h['images'])
                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))

                    with open(os.path.join(data_train_path, "label" + '.json'), 'r') as j:
                        self.labels.extend(json.load(j))

            elif flag == "test":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # with open(os.path.join(data_train_path,  "image_test"+ '.json'), 'r') as j:
                    #     self.imgs.extend(json.load(j))
                    self.h = h5py.File(os.path.join(data_train_path,  'image_processed_test' + '.hdf5'), 'r')
                    self.imgs.extend(self.h['images'])
                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        elif F == "OOD":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    self.h = h5py.File(os.path.join(data_train_path,  'image_processed' + '.hdf5'), 'r')
                    self.imgs.extend(self.h['images'])

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))


                    # Load encoded captions (completely into memory)
                    self.h = h5py.File(os.path.join(data_train_path,  'image_processed_test' + '.hdf5'), 'r')
                    self.imgs.extend(self.h['images'])

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.imgs)

        # print("client data:", self.dataset_size)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        # img =  "/home/ray/preject/data/wikipedia_class/sum_image/"  + self.imgs[i] + ".jpg"

        # print("1", img.shape)
        if self.transform is not None:

            img = Image.fromarray(np.uint8(self.imgs[i].transpose(1,2,0)))

            img = self.transform(img)

        caption = torch.FloatTensor(self.captions[i])
        if self.F == "N" and self.flag == "train":
            label = self.labels[i]
        else:
            label = 1
        return img, caption, label


    def __len__(self):
        return self.dataset_size









class wikipedia_read_clip(Dataset):

    def __init__(self, data_folder, flag, user_dict, user_id,F="N", LG = 0):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        data_distribution = user_dict[user_id]
        # Captions per image
        self.imgs = []
        self.captions = []
        self.labels = []
        self.F = F
        self.flag = flag

        if F == "N":
            if flag == "train":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image" + '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))

                    with open(os.path.join(data_train_path, "label" + '.json'), 'r') as j:
                        self.labels.extend(json.load(j))

            elif flag == "test":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    with open(os.path.join(data_train_path,  "image_test"+ '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        elif F == "OOD":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image"  + '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))


                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image_test"+ '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        # PyTorch transformation pipeline for the image (normalizing, etc.)

        # Total number of datapoints
        self.dataset_size = len(self.imgs)

        # print("client data:", self.dataset_size)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img =  "/home/ray/preject/data/wikipedia_class/sum_image/"  + self.imgs[i] + ".jpg"



        img = cv2.imread(img)
        img = Image.fromarray(np.uint8(img))
        image = preprocess(img).unsqueeze(0)
        img_feature = model_clip.encode_image(image)

        caption = torch.FloatTensor(self.captions[i])

        if self.F == "N" and self.flag == "train":
            label = self.labels[i]
        else:
            label = 1
        return img_feature.squeeze(), caption, label


    def __len__(self):
        return self.dataset_size




