# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:54:49 2020

@author: 709
"""
import tensorflow as tf
from keras.applications import InceptionV3
import numpy as np
from keras.applications import ResNet50,InceptionResNetV2,DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.optimizers import RMSprop,Adadelta
from keras import activations
from keras.models import load_model,Model
from keras import callbacks
from keras import regularizers
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import os
import datetime
from keras.losses import categorical_crossentropy
from keras.models import load_model
# import sys
# sys.path.append(r"C:\Users\709\Desktop\DenseNet-Keras-master")
# import  densenet121

path='../../cropdataset/'
path2='/stu11/grapedata/models/classification/DenseNet121/'
import os 
if not os.path.exists(path2):
    os.makedirs(path2)

x_train=np.load(path+'x1_train.npy')
y_train=np.load(path+'y1_train.npy')
x_validata=np.load(path+'x1_validata.npy')
y_validata=np.load(path+'y1_validata.npy')
#DenseNet=DenseNet121(weights='imagenet',include_top=False,input_shape=(196,196,3))
DenseNet=DenseNet121(weights=None,include_top=False,input_shape=(196,196,3))
DenseNet.trainable=True
# for layer in DenseNet.layers:
#     if layer.name=='conv2_block2_1_conv':
#         layer.trainable=True
#         print(layer.name+"is trainable")
#     if layer.name=='conv1/conv':
#         layer.trainable=True
#         print(layer.name+"is trainable")
#     if layer.name=='conv3_block3_1_conv':
#         layer.trainable=True
#         print(layer.name+"is trainable")
#     if layer.name=='conv2_block3_1_conv':
#         layer.trainable=True
#         print(layer.name+"is trainable")
plot_model(DenseNet,path2+'model_DenseNet121.png',show_layer_names=True)
input_=layers.Input((196,196,3))
x=DenseNet(input_)
x=layers.Flatten()(x)
x=layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.00005))(x)
x=layers.Dropout(0.5)(x)
output=layers.Dense(3,activation='softmax')(x)
model=Model(input_, output)
model.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['acc'])
history=model.fit(x_train,y_train,batch_size=16 ,epochs=200,validation_data=(x_validata,y_validata))
history=history.history
acc=history['acc']
loss=history['loss']
val_acc=history['val_acc']
val_loss=history['val_loss']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'r',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
max_val_acc_index=np.argmax(val_acc)
plt.plot(max_val_acc_index+1,val_acc[max_val_acc_index],'ks')
show_max='['+str(max_val_acc_index)+','+str(format(val_acc[max_val_acc_index],'.2f'))+']'
plt.annotate(show_max, xytext=(-20,-30),xy=(max_val_acc_index+1,val_acc[max_val_acc_index]),textcoords='offset points',arrowprops=dict(arrowstyle='->'))
plt.xlabel('Epochs')
plt.ylabel('Accuarcy')
plt.title('Training and validation accuracy of DenseNet-121 on group data (b)')
plt.legend(loc=3)
plt.grid(axis="y",linewidth=1)
plt.grid(axis="x",linewidth=1)
plt.savefig(path2+'DenseNet-121_acc(b).tif')
plt.savefig(path2+'DenseNet-121_acc(b).png')
plt.clf()
plt.plot(epochs,loss,'r',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and validation loss of DenseNet-121 on group data (b)')
plt.legend(loc=2)
plt.grid(axis="y",linewidth=1)
plt.grid(axis="x",linewidth=1)
plt.savefig(path2+'DenseNet-121_loss(b).tif')
plt.savefig(path2+'DenseNet-121_loss(b).png')
model.save(path2+'model_DenseNet121.h5')