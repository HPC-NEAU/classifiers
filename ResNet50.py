# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 15:19:21 2020

@author: VULCAN
"""
import numpy as np
from keras.applications import ResNet50,InceptionResNetV2
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

path='../../cropdataset/'
path2='/stu11/grapedata/models/classification/ResNet50/'
import os 
if not os.path.exists(path2):
    os.makedirs(path2)
    
x_train=np.load(path+'x1_train.npy')
y_train=np.load(path+'y1_train.npy')
x_test=np.load(path+'x1_validata.npy')
y_test=np.load(path+'y1_validata.npy')


input_=layers.Input((196,196,3))

res50=ResNet50(weights='imagenet',
                include_top=False,
                input_shape=(196,196,3))
'''
res50=ResNet50(weights='imagenet',
                include_top=False,
                input_shape=(196,196,3))
'''
res50.trainable=True#221 321 331
# for layer in res50.layers:
#     if layer.name=='conv1_conv':
#         layer.trainable=True
#         print(layer.name+" is trainable")
#     # if layer.name=='conv2_block1_0_conv':
#     #     layer.trainable=True
#     #     print(layer.name+' is trainable')
#     if layer.name=='conv2_block2_1_conv':
#         layer.trainable=True
#         print(layer.name+' is trainable')
#     # if layer.name=='conv2_block1_1_relu':
#     #     layer.trainable=True
#     #     print(layer.name+' is trainable')
#     if layer.name=='conv3_block2_1_conv':
#         layer.trainable=True
#         print(layer.name+' is trainable')
#     # if layer.name=='conv2_block1_2_relu':
#     #     layer.trainable=True
#     #     print(layer.name+' is trainable')
#     if layer.name=='conv3_block3_1_conv':
#         layer.trainable=True
#         print(layer.name+' is trainable')
x=res50(input_)
# res50.summary()
x=layers.Flatten()(x)
x=layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(x) #l2范数正则化，系数0.002，没有过拟合，最后5次准确率91%+-1.5%
x=layers.Dropout(0.4)(x)
output=layers.Dense(3,activation='softmax')(x)
model=Model(input_,output)
model.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['acc'])# 等sgd 训练后，尝试Adadelta以及牛顿动量法
#res50.summary()
plot_model(res50,to_file=path2+"model_res50.png",show_shapes=True,show_layer_names=True,rankdir="TB")
history=model.fit(x_train,y_train,batch_size=16,epochs=200,validation_data=(x_test,y_test))
history=history.history
acc=history['acc']
loss=history['loss']
val_acc=history['val_acc']
val_loss=history['val_loss']
# np.save('E:/apple/resnet/acc.npy',acc)
# np.save('E:/apple/resnet/val_acc.npy',val_acc)
# np.save('E:/apple/resnet/loss.npy',loss)
# np.save('E:/apple/resnet/val_loss.npy',val_loss)
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'r',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
max_val_acc_index=np.argmax(val_acc)
plt.plot(max_val_acc_index+1,val_acc[max_val_acc_index],'ks')
show_max='['+str(max_val_acc_index)+','+str(format(val_acc[max_val_acc_index],'.2f'))+']'
plt.annotate(show_max, xytext=(-40,-30),xy=(max_val_acc_index+1,val_acc[max_val_acc_index]),textcoords='offset points',arrowprops=dict(arrowstyle='->'))
plt.xlabel('Epochs')
plt.ylabel('Accuarcy')
plt.title('Training and validation accuracy of ResNet-50 on group data (b)')
plt.legend(loc=3)
plt.grid(axis="y",linewidth=1)
plt.grid(axis="x",linewidth=1)
plt.savefig(path2+'ResNet-50_acc(b).tif')
plt.savefig(path2+'ResNet-50_acc(b).png')
plt.clf()
plt.plot(epochs,loss,'r',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and validation loss of ResNet-50 on group data (b)')
plt.legend(loc=2)
plt.grid(axis="y",linewidth=1)
plt.grid(axis="x",linewidth=1)
plt.savefig(path2+'ResNet-50_loss(b).tif')
plt.savefig(path2+'ResNet-50_loss(b).png')
model.save(path2+'model_res50.h5')
#     weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH, cache_subdir='models')
#     model.load_weights(weights_path)

#     img_path = 'elephant.jpg'
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     print('Input image shape:', x.shape)

#     preds = model.predict(x)
#     print('Predicted:', decode_predictions(preds))
