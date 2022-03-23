# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 22:41:02 2020

@author: 709
"""
import scipy.misc
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image

path='../cropdataset'
y=np.load(path+'/y1.npy') #读入.npy文件
x=np.load(path+'/x1.npy')

print("type:", type(x))
print("shape:", x.shape)
print("type:", type(y))
print("shape:", y.shape)
#print("data:", data)




print(x.shape)
print(x[1].shape)

# for i in range(x.shape[0]): #读取原图
#     B=x[i, ;, ;, 0]
#     scipy.misc.imsave("path/im"+str(i)+"x"+".png",B) #保存为png格式，也可将png换位jpg等其他格式
x_train, x_validata, y_train, y_validata = train_test_split(x, y, test_size=0.2, stratify=y)
# x_train, x_validata_test, y_train, y_validata_test = train_test_split(x, y, test_size=0.4, stratify=y)
# x_validata, x_test, y_validata, y_test = train_test_split(x_validata_test, y_validata_test, test_size=0.5, stratify=y_validata_test)

print(x_train.shape)
# print(x_test.shape)
print(x_validata.shape)  
np.save(path+'/x1_train.npy',x_train)
np.save(path+'/y1_train.npy',y_train)
# np.save(path+'/x_test.npy',x_test)
# np.save(path+'/y_test.npy',y_test)
np.save(path+'/x1_validata.npy',x_validata)  # ===============
np.save(path+'/y1_validata.npy',y_validata)  # ===============
print(y_train.shape)
# print(y_test.shape)
print(y_validata.shape)

# y=to_categorical(y,num_classes=5)
# np.save('D:/AI CHALLENGER _dataset/apple/data/y_.npy',y)
