import os
import numpy as np
from skimage.filters import gaussian as gblur
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from MSCOCO_split import *
from torch.utils.data import random_split
# ref: https://github.com/HobbitLong/SupContrast
class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        # print("2", self.transform(x).shape)
        return [self.transform(x), self.transform(x)]


class TwoCropTransform_text:
    """Create two crops of the same image"""

    def __init__(self):
        self.transform = None

    def __call__(self, x):
        # print("2", self.transform(x).shape)

        return [x, add_noise(x)]

class OneCropTransform_text:
    """Create two crops of the same image"""

    def __init__(self):
        self.transform = None

    def __call__(self, x):
        # print("2", self.transform(x).shape)

        return x
def add_noise(x):

    x = x +  torch.randn(x.shape)
    return x
    




def gaussian(
    data_dir, batch_size, mode="base", normalize=True, norm_layer=None, size=32
):
    """
    Minimal version since we use this dataset only for OOD evaluation.
    """
    dummy_targets = torch.ones(10000)
    ood_data = torch.from_numpy(
        np.float32(
            np.clip(
                np.random.normal(loc=0.5, size=(10000, 3, 32, 32), scale=0.25), 0, 1
            )
        )
    )
    ood_data = torch.cat([norm_layer(x).unsqueeze(0) for x in ood_data])
    dataset = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return 0, loader, 0


def uniform(
    data_dir, batch_size, mode="base", normalize=True, norm_layer=None, size=32
):
    """
    Minimal version since we use this dataset only for OOD evaluation.
    """
    dummy_targets = torch.ones(10000)
    ood_data = torch.from_numpy(
        np.float32(np.clip(np.random.uniform(size=(10000, 3, 32, 32)), 0, 1))
    )
    ood_data = torch.cat([norm_layer(x).unsqueeze(0) for x in ood_data])
    dataset = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return 0, loader, 0


def MSCOCO_image( mode,   data_dir, user_dict, batch_size, normalize=True, norm_layer=None, size=256, data_name="coco_5_cap_per_img_5_min_word_freq", F="N"):

    # transform_train = [
    #     transforms.Resize(size),
    #     transforms.RandomCrop(size, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    # ]
    transform_test = [transforms.Resize(size), transforms.ToTensor()]

    if mode == "base":
        transform_train = [transforms.Resize(size), transforms.ToTensor()]
    elif mode == "ssl":
        transform_train = [
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]


    if norm_layer is None:
        norm_layer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    
    if normalize:
        transform_train.append(norm_layer)
        transform_test.append(norm_layer)

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    if mode == "ssl":
        transform_train = TwoCropTransform(transform_train)

    if F == "OOD":
        trainset = non_iid_MSCOCODataset_image(data_dir, data_name,'train', user_dict, 1, transform_train)
        
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = None
    else:
        trainset = non_iid_MSCOCODataset_image(data_dir, data_name,'train', user_dict, 0, transform_train)
        testset = non_iid_MSCOCODataset_image(data_dir, data_name, 'test', user_dict, 0, transform_test)

        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

    return train_loader, test_loader, norm_layer



def MSCOCO_image_with_CMA( mode,   data_dir, user_dict, batch_size, normalize=True, norm_layer=None, size=256, data_name="coco_5_cap_per_img_5_min_word_freq", F="N", FineGrain=0):

    transform_train = [
        transforms.Resize(size),
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    transform_test = [transforms.Resize(size), transforms.ToTensor()]

    if mode == "base":
        transform_train = [transforms.Resize(size), transforms.ToTensor()]
    elif mode == "ssl":
        transform_train = [
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]


    if norm_layer is None:
        norm_layer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )



    if normalize:
        transform_train.append(norm_layer)
        transform_test.append(norm_layer)



        
    # transform_test = [transforms.Resize(size), transforms.ToTensor()]
    # transform_train = [transforms.Resize(size), transforms.ToTensor()]



    

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    if mode == "ssl":
        transform_train = TwoCropTransform(transform_train)

    if F == "OOD":
        trainset = non_iid_MSCOCODataset(data_dir, data_name,'train', user_dict, 1, transform_train)
        
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = None
    else:
        trainset = non_iid_MSCOCODataset(data_dir, data_name,'train', user_dict, 0, transform_train)
        testset = non_iid_MSCOCODataset(data_dir, data_name, 'test', user_dict, 0, transform_test)

        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

    return train_loader, test_loader, norm_layer













def MSCOCO_image_with_CMG_CMER( mode,   data_dir, user_dict, batch_size, normalize=True, norm_layer=None, size=256, data_name="coco_5_cap_per_img_5_min_word_freq", F="N",FineGrain=0):


    transform_test = [transforms.Resize(size), transforms.ToTensor()]

    if mode == "base":
        transform_train = [  transforms.Resize(size), transforms.ToTensor()]
    elif mode == "ssl":
        transform_train = [
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]


    if norm_layer is None:
        norm_layer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    
    if normalize:
        transform_train.append(norm_layer)
        transform_test.append(norm_layer)

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    if mode == "ssl":
        transform_train = TwoCropTransform(transform_train)

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
            testset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

    return train_loader, test_loader, norm_layer









def UCM_caption( mode,   data_dir, user_dict, batch_size, normalize=True, norm_layer=None, size=32, F="N", LG = 0):


    # transform_train = [
    #     transforms.Resize(size),
    #     transforms.RandomCrop(size, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    # ]
    transform_test = [transforms.Resize(size), transforms.ToTensor()]

    if mode == "base":
        transform_train = [transforms.Resize(size), transforms.ToTensor()]
    elif mode == "ssl":
        transform_train = [
            transforms.Resize(size),
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]


    # if norm_layer is None:
    #     norm_layer = transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #     )
    
    # if normalize:
    #     transform_train.append(norm_layer)
    #     transform_test.append(norm_layer)

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    if mode == "ssl":
        transform_train = TwoCropTransform(transform_train)


    if F == "OOD":
        trainset = UCM_caption_LG(data_dir, 'train', user_dict, 1, transform_train,F, LG)
        
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = None
    else:
        trainset = UCM_caption_LG(data_dir,'train', user_dict, 0, transform_train, F, LG )
        testset = UCM_caption_LG(data_dir,  'test', user_dict, 0, transform_test, F, LG)

        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

    return train_loader, test_loader, norm_layer






def UCM_caption_raw( mode,   data_dir, user_dict, batch_size, normalize=True, norm_layer=None, size=32, F="N", LG = 0):


    # transform_train = [
    #     transforms.Resize(size),
    #     transforms.RandomCrop(size, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    # ]
    transform_test = [transforms.Resize(size), transforms.ToTensor()]

    if mode == "base":
        transform_train = [transforms.Resize(size), transforms.ToTensor()]
    elif mode == "ssl":
        transform_train = [
            transforms.Resize(size),
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]


    # if norm_layer is None:
    #     norm_layer = transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #     )
    
    # if normalize:
    #     transform_train.append(norm_layer)
    #     transform_test.append(norm_layer)

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    if mode == "ssl":
        transform_train = TwoCropTransform(transform_train)


    if F == "OOD":
        trainset = UCM_caption_read(data_dir, 'train', user_dict, 1, transform_train,F, LG)
        
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = None
    else:
        trainset = UCM_caption_read(data_dir,'train', user_dict, 0, transform_train, F, LG )
        testset = UCM_caption_read(data_dir,  'test', user_dict, 0, transform_test, F, LG)

        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

    return train_loader, test_loader, norm_layer








def UCM_caption_I2T( mode,   data_dir, user_dict, batch_size, normalize=True, norm_layer=None, size=32, F="N", LG = 0):


    transform_test = [transforms.Resize(size), transforms.ToTensor()]
    transform_train = [transforms.Resize(size), transforms.ToTensor()]
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

 

    if F == "OOD":
        trainset = UCM_caption_detection_I2T(data_dir, 'train', user_dict, 1, transform_train,F, LG)
        
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = None
    else:
        trainset = UCM_caption_detection_I2T(data_dir,'train', user_dict, 0, transform_train, F, LG )
        testset = UCM_caption_detection_I2T(data_dir,  'test', user_dict, 0, transform_test, F, LG)

        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

    return train_loader, test_loader, norm_layer








def wikipedia( mode,   data_dir, user_dict, batch_size, normalize=True, norm_layer=None, size=32, F="N", LG = 0):



    transform_test = [transforms.Resize((size,size)), transforms.ToTensor()]

    if mode == "base":
        transform_train = [transforms.Resize((size,size)), transforms.ToTensor()]
    elif mode == "ssl":
        transform_train = [
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]

    # transform_train = [transforms.Resize((size,size)), transforms.ToTensor()]
    
    norm_layer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform_train.append(norm_layer)
    transform_test.append(norm_layer)

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    if mode == "ssl":
        transform_train = TwoCropTransform(transform_train)


    if F == "OOD":
        trainset = wikipedia_read(data_dir, 'train', user_dict, 1, transform_train,F, LG)
        
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = None
    else:
        trainset = wikipedia_read(data_dir,'train', user_dict, 0, transform_train, F, LG )
        testset = wikipedia_read(data_dir,  'test', user_dict, 0, transform_test, F, LG)

        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

    return train_loader, test_loader, norm_layer



def wikipedia_CMER( mode,   data_dir, user_dict, batch_size, normalize=True, norm_layer=None, size=32, F="N", LG = 0):



    transform_test = [transforms.Resize((size,size)), transforms.ToTensor()]

    if mode == "base":
        transform_train = [transforms.Resize((size,size)), transforms.ToTensor()]
    elif mode == "ssl":
        transform_train = [
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]

    # transform_train = [transforms.Resize((size,size)), transforms.ToTensor()]
    
    norm_layer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform_train.append(norm_layer)
    transform_test.append(norm_layer)

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    if mode == "ssl":
        transform_train = TwoCropTransform(transform_train)


    if F == "OOD":
        trainset = wikipedia_LG(data_dir, 'train', user_dict, 1, transform_train,F, LG)
        
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = None
    else:
        trainset = wikipedia_LG(data_dir,'train', user_dict, 0, transform_train, F, LG )
        testset = wikipedia_LG(data_dir,  'test', user_dict, 0, transform_test, F, LG)

        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

    return train_loader, test_loader, norm_layer





def wikipedia_I2T( mode,   data_dir, user_dict, batch_size, normalize=True, norm_layer=None, size=32, F="N", LG = 0):



    transform_test = [transforms.Resize((size,size)), transforms.ToTensor()]
    transform_train = [transforms.Resize((size,size)), transforms.ToTensor()]



    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)



    if F == "OOD":
        trainset = wikipedia_read(data_dir, 'train', user_dict, 1, transform_train,F, LG)
        
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = None
    else:
        trainset = wikipedia_read(data_dir,'train', user_dict, 0, transform_train, F, LG )
        testset = wikipedia_read(data_dir,  'test', user_dict, 0, transform_test, F, LG)

        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

    return train_loader, test_loader, norm_layer








