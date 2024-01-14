import os
import numpy as np
from skimage.filters import gaussian as gblur
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
import h5py
import json
from PIL import Image
import copy
import cv2
import clip
class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        # print("2", self.transform(x).shape)
        
        return [self.transform(x), self.transform(x)]


def MSCOCO_image_with_CMA_D( mode,   data_dir, user_dict, batch_size, normalize=True, norm_layer=None, size=256, data_name="coco_5_cap_per_img_5_min_word_freq", F="N",FineGrain=0):


    transform_test = [ transforms.ToTensor()]
    transform_train = [ transforms.ToTensor()]
    # if mode == "base":
    #     transform_train = [transforms.Resize(size), transforms.ToTensor()]
    # elif mode == "ssl":
    #     transform_train = [
    #         transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    #         transforms.RandomGrayscale(p=0.2),
    #         transforms.ToTensor(),
    #     ]


    # if norm_layer is None:
    #     norm_layer = transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #     )
    
    # if normalize:
    #     transform_train.append(norm_layer)
    #     transform_test.append(norm_layer)

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)


    if F == "OOD":
        trainset = non_iid_MSCOCODataset_Distillation(data_dir, data_name,'train', user_dict, 1, transform_train, FineGrain)
        
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = None
    else:
        trainset = non_iid_MSCOCODataset_Distillation(data_dir, data_name,'train', user_dict, 0, transform_train,FineGrain)
        testset = non_iid_MSCOCODataset_Distillation(data_dir, data_name, 'test', user_dict, 0, transform_test,FineGrain)

        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

    return train_loader, test_loader, norm_layer




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
        self.max = 43.5078
        self.min = 14.5456
        if flag == "train":
            for i in data_distribution:
                data_train_path = data_folder + str(i)

                # Open hdf5 file where images are stored
                if FineGrain == 1:
                    self.h = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name + 'clip_processed_1to1.hdf5'), 'r')
                    # self.imgs.extend(self.h['images'])
                    self.probs.extend(self.h['prob'])
                    self.H = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name + '.hdf5'), 'r')
                    self.imgs.extend(self.H['images'])
                else:
                    self.h = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name + '.hdf5'), 'r')
                    self.imgs.extend(self.h['images'])                    
                # Load encoded captions (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPTIONS_' + data_name + "vector" + '.json'), 'r') as j:
                    self.captions.extend(json.load(j))
            
    


                # with open(os.path.join(data_train_path,  '_CAPTIONS_' + data_name  + '.json'), 'r') as j:
                #     self.captions_words.extend(json.load(j))

                # Load caption lengths (completely into memory)
                # with open(os.path.join(data_train_path,  '_CAPLENS_' + data_name + '.json'), 'r') as j:
                #     self.caplens.extend(json.load(j))
        elif flag == "test":
            for i in data_distribution:
                data_train_path = data_folder + str(i)

                # Open hdf5 file where images are stored


                if FineGrain == 1:
                    self.h = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name + 'clip_processed_1to1_test.hdf5'), 'r')
                    # self.imgs.extend(self.h['images'])
                    self.probs.extend(self.h['prob'])
                    self.H = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name+ "_test" + '.hdf5'), 'r')
                    self.imgs.extend(self.H['images'])
                else:
                    self.h = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name+ "_test" + '.hdf5'), 'r')
                    self.imgs.extend(self.h['images'])                    


                # Load encoded captions (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPTIONS_' + data_name + "_test" + "vector" + '.json'), 'r') as j:
                    self.captions.extend(json.load(j))

                # with open(os.path.join(data_train_path,  '_CAPTIONS_' + data_name + "_test"  + '.json'), 'r') as j:
                #     self.captions_words.extend(json.load(j))

                # Load caption lengths (completely into memory)
                # with open(os.path.join(data_train_path,  '_CAPLENS_' + data_name + "_test" + '.json'), 'r') as j:
                #     self.caplens.extend(json.load(j))
            
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

        # print("client data:", self.dataset_size)

    def __getitem__(self, i):

        if self.FineGrain == 1:
            model_clip, preprocess = clip.load("ViT-B/32", device = "cpu")
            img = self.imgs[i//5]
            # print(img.shape)
            (depth, width, length ) = img.shape

            # img = Image.fromarray(np.uint8(img.transpose(1,2,0)))


            
            feature_list = []
            cut_width = int(width / 2)
            cut_length = int(length / 2)
            for w in range(0, 2):
                for j in range(0, 2):
                    temp_img = copy.deepcopy(img)
                    pic = torch.zeros((depth, cut_width, cut_length))
                    temp_img[:,w*cut_width : (w+1)*cut_width, j*cut_length : (j+1)*cut_length] = pic
    
                    pic = Image.fromarray(np.uint8(temp_img.transpose(1,2,0) ))

                    image = preprocess( pic ).unsqueeze(0)
                    with torch.no_grad():
                        image_features = model_clip.encode_image(image)
                        # text_features = model_clip.encode_text(text)
                        feature_list.append(image_features.squeeze(0))


            img = feature_list
            # probs = (torch.FloatTensor(self.probs[i]) - self.min) / (self.max - self.min)
            probs = torch.FloatTensor(self.probs[i]) 
        else: 
            img = self.imgs[i//5]
            img = Image.fromarray(np.uint8(img.transpose(1,2,0)))
            img = self.transform(img)
            probs = torch.tensor([])
        # print(len(self.captions[i]))
        caption =  torch.FloatTensor(self.captions[i])

        # caplen = torch.FloatTensor([self.caplens[i]])

        
        return img, caption, probs


    def __len__(self):
        return self.dataset_size

    def get_len(slef):
        return   self.dataset_size






def Flower_detection_with_CMA( mode,   data_dir, user_dict, batch_size, normalize=True, norm_layer=None, size=32, F="N"):


    transform_train = [
        transforms.Resize(size),
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    transform_test = [transforms.Resize((size,size)), transforms.ToTensor()]

    if mode == "base":
        transform_train = [transforms.Resize((size,size)), transforms.ToTensor()]
    elif mode == "train":
        transform_train = [
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]
    # transform_test = [transforms.Resize((size,size)), transforms.ToTensor()]
    # transform_train = [transforms.Resize((size,size)), transforms.ToTensor()]

    if norm_layer is None:
        norm_layer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    
    if normalize:
        transform_train.append(norm_layer)
        transform_test.append(norm_layer)

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)


    if F == "OOD":
        trainset = Flower_detection(data_dir, 'train', user_dict, 1, transform_train,F)
        
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = None
    else:
        trainset = Flower_detection(data_dir,'train', user_dict, 0, transform_train)
        testset = Flower_detection(data_dir,  'test', user_dict, 0, transform_test)

        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

    return train_loader, test_loader, norm_layer




class Flower_detection(Dataset):

    def __init__(self, data_folder, flag, user_dict, user_id,transform=None,F="N"):
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
        self.dataset_size = len(self.captions)

        # print("client data:", self.dataset_size)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = "/home/an/project/data/Flower_102/jpg/" + self.imgs[i][-15:] 

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
        caption =  torch.FloatTensor(self.captions[i][0])

        return img, caption


    def __len__(self):
        return self.dataset_size







def UCM_caption_I2T( mode,   data_dir, user_dict, batch_size, normalize=True, norm_layer=None, size=256, data_name="coco_5_cap_per_img_5_min_word_freq", F="N",FineGrain=0):


    transform_test = [transforms.Resize(size), transforms.ToTensor()]
    transform_train = [transforms.Resize(size), transforms.ToTensor()]
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

 

    if F == "OOD":
        trainset = UCM_caption_detection_I2T(data_dir, 'train', user_dict, 1, transform_train)

        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = None
    else:
        trainset = UCM_caption_detection_I2T(data_dir,'train', user_dict, 0, transform_train)
        testset = UCM_caption_detection_I2T(data_dir,  'test', user_dict, 0, transform_test)

        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

    return train_loader, test_loader, norm_layer




class UCM_caption_detection_I2T(Dataset):

    def __init__(self, data_folder,  flag, user_dict, user_id, transform):
   
        data_distribution = user_dict[user_id]
        # Captions per image

        self.cpi = 1
        self.imgs = []
        self.imgs0 = []
        self.imgs1 = []
        self.imgs2 = []
        self.imgs3 = []

        self.probs = []
        self.caplens = []
        self.captions = []
        self.captions_words = []
        
        if flag == "train":
            for i in data_distribution:
                data_train_path = data_folder + str(i)
  

                self.h = h5py.File(os.path.join(data_train_path,  'image_processed.hdf5'), 'r')
                self.probs.extend(self.h['prob'])
                self.imgs0.extend(self.h['images0'])
                self.imgs1.extend(self.h['images1'])
                self.imgs2.extend(self.h['images2'])
                self.imgs3.extend(self.h['images3'])

    
        elif flag == "test":
            for i in data_distribution:
                data_train_path = data_folder + str(i)

                self.h = h5py.File(os.path.join(data_train_path,  'image_processed_test.hdf5'), 'r')
                # self.imgs.extend(self.h['images'])
                self.probs.extend(self.h['prob'])
                # self.H = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name+ "_test" + '.hdf5'), 'r')
                # self.imgs.extend(self.H['images'])
                self.imgs0.extend(self.h['images0'])
                self.imgs1.extend(self.h['images1'])
                self.imgs2.extend(self.h['images2'])
                self.imgs3.extend(self.h['images3'])
        self.dataset_size = len(self.imgs0)
    def __getitem__(self, i):
        img = [self.imgs0[i], self.imgs1[i], self.imgs2[i], self.imgs3[i]]
        probs = torch.FloatTensor(self.probs[i]) 
        return img,  probs


    def __len__(self):
        return self.dataset_size

