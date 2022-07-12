import os
from os import walk
import glob
import numpy as np
import cv2
from tqdm import tqdm
import copy
from sklearn.decomposition import PCA
import sys
import random
sys.setrecursionlimit(30000)
print("Python Recursive Limitation = ", sys.getrecursionlimit())

root = '/home/admin1/Documents/Atik/Meta_Learning/MAML-Pytorch/datasets/256'
mood = 'train'
path = os.path.join(root, mood, '')  

def load_files(path: str) -> list:
    
    filenames = next(walk(path))[1]
    dictLabels = {}
    for i in range(len(filenames)):  
        img = []
        for images in glob.iglob(f'{path+filenames[i]}/*'):
            # check if the image ends with png
            if (images.endswith(".jpg")):
                img_temp = images[len(path+filenames[i]+'/'):]
                img_temp = filenames[i]+'/'+img_temp
                img.append(img_temp)
            
            dictLabels[filenames[i]] = img

    return list(dictLabels.values())

file = load_files(path)   

classes = copy.deepcopy(file)
class_num = len(classes)

images = []
for k in tqdm(range(class_num)):
    images_n = [cv2.imread(path+classes[k][j]) for j in range(len(classes[k]))]    
    images.append(images_n)

cluster_num = 10

neg_clusters = []
neg_images = []

with tqdm(total=len(classes)*cluster_num*len(file[0])) as pbar:
    for current_class in tqdm(range(len(classes))):
        file_new = copy.deepcopy(file)
        random.shuffle(file_new)
        class_array_cluster = []
        image_array_cluster = []
        for j in tqdm(range(cluster_num)):
            class_array = []
            image_temp = []
            for i in range(len(file[0])):
                rand_class = int(np.random.randint(len(file_new),size=(1, 1)))
                
                while rand_class == current_class or len(file_new[rand_class]) == 0:
                    rand_class = int(np.random.randint(len(file_new),size=(1, 1)))
                
                rand_image = int(np.random.randint(len(file_new[rand_class]),size=(1, 1)))
        
                file_array = file_new[rand_class][rand_image]
                file_new[rand_class].remove(file_new[rand_class][rand_image])
                class_array.append(file_array)
                image_temp.append(cv2.imread(path+file_array))
                pbar.update(1)
                
            class_array_cluster.append(class_array)
            image_array_cluster.append(image_temp)
            
        neg_clusters.append(class_array_cluster)
        neg_images.append(image_array_cluster)


def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def rep_image(images: list, classes: list) -> list:
    
    indexes = []
    for j in tqdm(range(len(images))):
    
        img2 =  np.array([images[j][i].mean(axis=2).flatten() for i in range(len(images[j]))])
        img2 = np.array(img2)
    
        norm_img2 = normalizeData(img2)
    
        pca = PCA(2)
    
        data = pca.fit_transform(norm_img2)
    
        centroid = np.expand_dims(data.mean(axis=0), axis = 0)
    
        dist = [np.linalg.norm(data[i] - centroid) for i in range(len(data))]
    
        index_min = np.argmin(dist)
        indexes.append(index_min)
        
        class_rep = [classes[i][indexes[i]] for i in range(len(indexes))]
    
    return class_rep

def orb_sim(img1, img2):
    
  orb = cv2.ORB_create()

  # detect keypoints and descriptors
  kp_a, desc_a = orb.detectAndCompute(img1, None)
  kp_b, desc_b = orb.detectAndCompute(img2, None)
  
  if desc_a is None or desc_b is None:
      return 0

  # define the bruteforce matcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
  #perform matches. 
  matches = bf.match(desc_a, desc_b)
  #Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
  similar_regions = [i for i in matches if i.distance < 50]  
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)

indx = rep_image(images, classes)

indx_neg = [rep_image(neg_images[i], neg_clusters[i]) for i in tqdm(range(len(neg_images)))]





