# -*- coding: utf-8 -*-
"""
@author: 709
"""

import os,glob
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf 
from tensorflow import keras 
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import random

path='../cropdataset'
data=['BlackMeaslesFungus', 'BlackRotFungus', 'LeafBlightFungus']

y=[]
x=[]

print(data)
print(len(data))

indexid=0
for val in data:
    filename = []
    for name in os.listdir(path + '/' + val):
        if 'jpg' not in name:
            continue
        filename.append(name)
    random.shuffle(filename)
    if len(filename) < 500:
        print("ERROR")
        break
        
    for i in range(500):
        y.append(indexid)
#         img = Image.open(path + '/' + val + '/' + filename[i])
        img=cv2.imread(path+'/'+val+'/'+filename[i], cv2.IMREAD_UNCHANGED)
        img=cv2.resize(img,(196,196),interpolation=cv2.INTER_CUBIC)
        img=np.array(img)
        x.append(img)
        #print(img.shape)
   
    indexid = indexid + 1
    print(indexid)
    x1=np.array(x)
    y1=np.array(y)
    print(x1.shape)
    print(y1.shape)

x=np.array(x)
y=np.array(y)
print(x.shape)
y=to_categorical(y,num_classes=len(data))
print(y.shape)

np.save(path+'/x1.npy',x)
np.save(path+'/y1.npy',y)


'''
for filename in os.listdir(path+'/'+data[0]):
    for picture in os.listdir(path+'/'+data[0]+'/'+filename):
        y.append(0)
        image=Image.open(path+'/'+data[0]+'/'+filename+'/'+picture)
        image=np.array(image)
        x.append(image)
        print(image.shape)
        
for picture in os.listdir(path+'/'+data[1]):
    y.append(1)
    image=Image.open(path+'/'+data[1]+'/'+picture)
    image=np.array(image)
    x.append(image)
    print(image.shape)


#读入图像 并统一图像尺寸
for filename in os.listdir(path+'/'+data[0]):
    for picture in os.listdir(path+'/'+data[0]+'/'+filename):
        y.append(0)
        #image=Image.open(path+'/'+data[0]+'/'+filename+'/'+picture)
        img=cv2.imread(path+'/'+data[0]+'/'+filename+'/'+picture, cv2.IMREAD_UNCHANGED)
        img=cv2.resize(img,(196,196),interpolation=cv2.INTER_CUBIC)
        img=np.array(img)
        x.append(img)
        print(img.shape)
      
for picture in os.listdir(path+'/'+data[1]):
    y.append(1)
    img=cv2.imread(path+'/'+data[1]+'/'+picture, cv2.IMREAD_UNCHANGED)
    img=cv2.resize(img,(196,196),interpolation=cv2.INTER_CUBIC)
    img=np.array(img)
    x.append(img)
    print(img.shape)


x=np.array(x)
y=np.array(y)
print(x.shape)
y=to_categorical(y,num_classes=len(data))
print(y.shape)


#np.save(path+'/x.npy',x)
#np.save(path+'/y.npy',y)

'''


#img = cv2.imread('./Pictures/python.png', cv2.IMREAD_UNCHANGED)

