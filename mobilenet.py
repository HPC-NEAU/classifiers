import warnings

import numpy as np
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import (Activation, BatchNormalization, Conv2D,
                          DepthwiseConv2D, Dropout, GlobalAveragePooling2D,
                          GlobalMaxPooling2D, Input, Reshape)
from keras.models import Model
from keras.preprocessing import image
from keras.utils.data_utils import get_file

BASE_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.6/'

#---------------------------------------#
#   激活函数 relu6
#---------------------------------------#
def relu6(x):
    return K.relu(x, max_value=6)

#---------------------------------------#
#   卷积块
#   卷积 + 标准化 + 激活函数
#---------------------------------------#
def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)

#---------------------------------------#
#   深度可分离卷积块
#   3x3深度可分离卷积 + 1x1普通卷积
#---------------------------------------#
def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1, 1), block_id=1):

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

def MobileNet(input_shape=[224,224,3],
              depth_multiplier=1,
              dropout=1e-3,
              classes=3):
    img_input = Input(shape=input_shape)

    # 224,224,3 -> 112,112,32  
    x = _conv_block(img_input, 32, strides=(2, 2))
    # 112,112,32 -> 112,112,64
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)


    # 112,112,64 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)


    # 56,56,128 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)
    

    # 28,28,256 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # 14,14,512 -> 7,7,1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    # 7,7,1024 -> 1,1,1024
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)

    # 1,1,1024 -> 1,1,num_classes
    x = Conv2D(classes, (1, 1),padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    inputs = img_input

    model = Model(inputs, x, name='mobilenet_1_0_224_tf')
    return model

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

import tensorflow as tf
import os
import datetime
dataset = 'Fungus'  # 数据集的名称
path=f'../images/{dataset}/'
path2='./'
import os 
if not os.path.exists(path2):
    os.makedirs(path2)


if __name__ == '__main__':
    x_train=np.load(path+'x_train.npy')
    y_train=np.load(path+'y_train.npy')
    x_test=np.load(path+'x_validata.npy')
    y_test=np.load(path+'y_validata.npy')

    strategy=tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = MobileNet(input_shape=(64, 64, 3))
    model.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['acc'])
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
    plt.title('Training and validation accuracy of ResNet-50 on group data (c)')
    plt.legend(loc=3)
    plt.grid(axis="y",linewidth=1)
    plt.grid(axis="x",linewidth=1)
    plt.savefig(path2+f'MobileNet_acc({dataset}).tif')
    plt.savefig(path2+f'MobileNet_acc({dataset}).png')
    plt.clf()
    plt.plot(epochs,loss,'r',label='Training loss')
    plt.plot(epochs,val_loss,'b',label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and validation loss of ResNet-50 on group data (c)')
    plt.legend(loc=2)
    plt.grid(axis="y",linewidth=1)
    plt.grid(axis="x",linewidth=1)
    plt.savefig(path2+f'MobileNet_loss({dataset}).tif')
    plt.savefig(path2+f'MobileNet_loss({dataset}).png')
    model.save(path2+f'ResNet50({dataset}).h5')

    x=np.load(path+'/'+'x_test.npy')
    y=np.load(path+'/'+'y_test.npy')
    model.evaluate(x,y)
    
    
    
#     model_name = 'mobilenet_1_0_224_tf.h5'
#     weigh_path = BASE_WEIGHT_PATH + model_name
#     weights_path = get_file(model_name, weigh_path, cache_subdir='models')
#     model.load_weights(weights_path)

#     img_path = 'elephant.jpg'
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     print('Input image shape:', x.shape)

#     preds = model.predict(x)
#     print('Predicted:', decode_predictions(preds, 1))
