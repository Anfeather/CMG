from bert_serving.client import BertClient
import sys
import time
import numpy as np
import os
from tqdm import tqdm
import json
# class non_iid_MSCOCODataset(Dataset):

#     def __init__(self, data_folder="/home/an/project/data/MS_COCO_non_IID/anomaly_detection/non_iid_MSCOCO_train_30_50/"):
#         """
#         :param data_folder: folder where data files are stored
#         :param data_name: base name of processed datasets
#         :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
#         :param transform: image transform pipeline
#         """
#         data_distribution = list(range(80))
#         # Captions per image
#         self.cpi = 5
#         self.imgs = []d
#         self.caplens = []
#         self.captions = []

#         for i in data_distribution:
#             data_train_path = data_folder + str(i)

#             # Open hdf5 file where images are stored


#             # Load encoded captions (completely into memory)
#             with open(os.path.join("/home/an/project/data/MS_COCO_non_IID/anomaly_detection/non_iid_MSCOCO_train_30_50/",  '_CAPTIONS_' + "coco_5_cap_per_img_5_min_word_freq" + '.json'), 'r') as j:
#                 self.captions.extend(json.load(j))

#             # Load caption lengths (completely into memory)
#             with open(os.path.join(data_train_path,  '_CAPLENS_' + "coco_5_cap_per_img_5_min_word_freq" + '.json'), 'r') as j:
#                 self.caplens.extend(json.load(j))
        


#         # Total number of datapoints
#         self.dataset_size = len(self.captions)

#         # print("client data:", self.dataset_size)

#     def __getitem__(self, i):
#         # Remember, the Nth caption corresponds to the (N // captions_per_image)th image


#         caption = torch.LongTensor(self.captions[i])

#         caplen = torch.LongTensor([self.caplens[i]])


#         return img, caption, caplen


    # def __len__(self):
    #     return self.dataset_size


if __name__ == '__main__':
    bc = BertClient(check_length=False)
    # print(len(bc.encode(['First do it', 'then do it right', 'then do it better'])[0]))

    # encode a list of strings
    train_ID = [ 5, 3, 71, 46, 50, 59, 7, 0, 16, 53, 77, 13, 15, 10, 8, 9, 75, 11, 17, 21, 18, 19, 61, 20, 6, 14, 22, 74, 4, 23]
    data_folder = "/home/an/project/data/MS_COCO_non_IID/anomaly_detection/non_iid_MSCOCO_train_30_50/"
    data_distribution = list(range(80))
    for i in tqdm(data_distribution):
        data_train_path = data_folder + str(i)
        captions = []
        with open(os.path.join(data_train_path,  '_CAPTIONS_' + "coco_5_cap_per_img_5_min_word_freq" + '.json'), 'r') as j:
            captions.extend(json.load(j))
        
        len_a = len(captions)
        captions = [sen[0] for sen in captions]
        len_b = len(captions)
        assert len_a == len_b
        # print(captions)
        # print(len_a)
        
        senvec = bc.encode(captions).tolist()

        with open(os.path.join(data_train_path,  '_CAPTIONS_' + "coco_5_cap_per_img_5_min_word_freq" +"vector" + '.json'), 'w') as j:
            json.dump(senvec, j)

        if i in train_ID:
            captions = []
            with open(os.path.join(data_train_path,  '_CAPTIONS_' + "coco_5_cap_per_img_5_min_word_freq" +"_test" + '.json'), 'r') as j:
                captions.extend(json.load(j))
            
            len_a = len(captions)
            captions = [sen[0] for sen in captions]
            len_b = len(captions)
            assert len_a == len_b
            # print(captions)
            # print(len_a)
            
            senvec = bc.encode(captions).tolist()

            with open(os.path.join(data_train_path,  '_CAPTIONS_' + "coco_5_cap_per_img_5_min_word_freq" +"_test" + "vector" + '.json'), 'w') as j:
                json.dump(senvec, j)

    # with open('/home/an/project/cross-model/detection/CMDA/README.md') as fp:
    #     data = [v for v in fp if v.strip()][:512]
    #     num_tokens = sum(len([vv for vv in v.split() if vv.strip()]) for v in data)

    # show_tokens = len(sys.argv) > 3 and bool(sys.argv[3])

    # data_array = np.array(bc.encode(data))
    # print(data_array.shape)
    # np.savetxt("senvec.txt",data_array)
    
    
