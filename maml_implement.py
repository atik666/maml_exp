import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random
from tqdm import tqdm
from os import walk
import glob
import pickle

batchsz = 20  # batch of set, not batch of imgs
n_way = 2  # n-way
k_shot = 5  # k-shot
k_query = 15  # for evaluation
setsz = n_way * k_shot  # num of samples per set
querysz = n_way * k_query  # number of samples per set for evaluation
resize = 84  # resize to
startidx = 0  # index label not from 0, but from startidx

transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                     transforms.Resize((resize, resize)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                     ])


with open("final_pos_classes", "rb") as fp:
    file = pickle.load(fp)

file_dict = {v: k for v, k in enumerate(file)}

with open("final_neg_classes", "rb") as fp:
    file_neg = pickle.load(fp)

file_neg_dict = {v+256: k for v, k in enumerate(file_neg)}

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

file_all = Merge(file_dict, file_neg_dict)

data = []
img2label = {}
for i, (label, imgs) in enumerate(file_all.items()):
    data.append(imgs)  # [[img1, img2, ...], [img111, ...]]
    img2label[label] = i + startidx  # {"img_name[:9]":label}
cls_num = len(data)


support_x_batch = []  # support set batch
query_x_batch = []  # query set batch
selected_classes = []
for b in range(batchsz):  # for each batch
    # 1.select n_way classes randomly
    selected_cls_pos = np.random.choice(int(cls_num/2), 1, False)  # no duplicate
    selected_cls_neg = selected_cls_pos+256 # no duplicate
    selected_cls = np.concatenate((selected_cls_pos, selected_cls_neg), axis=0)
    #np.random.shuffle(selected_cls)
    support_x = []
    query_x = []
    selected_classes_temp = []
    for cls in selected_cls:
        # 2. select k_shot + k_query for each class
        selected_imgs_idx = np.random.choice(len(data[cls]), k_shot + k_query, False)
        np.random.shuffle(selected_imgs_idx)
        indexDtrain = np.array(selected_imgs_idx[:k_shot])  # idx for Dtrain
        indexDtest = np.array(selected_imgs_idx[k_shot:])  # idx for Dtest
        support_x.append(
            np.array(data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
        query_x.append(np.array(data[cls])[indexDtest].tolist())
        selected_classes_temp.append(cls)

    # shuffle the correponding relation between support set and query set
    random.shuffle(support_x)
    random.shuffle(query_x)

    support_x_batch.append(support_x)  # append set to current sets
    query_x_batch.append(query_x)  # append sets to current sets
    selected_classes.append(selected_classes_temp)


for index in range(batchsz):

    support_y_list = []
    for i in range(len(support_x_batch[index])):
        class_temp = np.repeat(selected_classes[index][i], len(support_x_batch[index][i]))
        support_y_list.append(class_temp)
    support_y = np.array(support_y_list).flatten().astype(np.int32)
    
    query_y_list = []
    for i in range(len(query_x_batch[index])):
        class_temp = np.repeat(selected_classes[index][i], len(query_x_batch[index][i]))
        query_y_list.append(class_temp)
    query_y = np.array(query_y_list).flatten().astype(np.int32)


unique = np.unique(support_y)
random.shuffle(unique)
# relative means the label ranges from 0 to n-way
support_y_relative = np.zeros(setsz)
query_y_relative = np.zeros(querysz)
for idx, l in enumerate(unique):
    print(idx, "", l)
    support_y_relative[support_y == l] = idx
    query_y_relative[query_y == l] = idx



















