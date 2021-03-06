import math

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import (Activation, Add, Conv2D, Dense, DepthwiseConv2D,
                          Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D,
                          Input, MaxPooling2D, ZeroPadding2D)
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
from keras.utils.data_utils import get_file
from tensorflow.keras import backend

BASE_WEIGHT_PATH = ('https://github.com/JonathanCMitchell/mobilenet_v2_keras/'
                    'releases/download/v1.1/')

# 用于计算padding的大小
def correct_pad(inputs, kernel_size):
    img_dim = 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

#---------------------------------------#
#   激活函数 relu6
#---------------------------------------#
def relu6(x):
    return K.relu(x, max_value=6)
    
#-----------------------------------------------------------#
#   使每层卷积的通道数可以被8整除，因为使用到了膨胀系数α
#-----------------------------------------------------------#
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

#-----------------------------#
#   逆瓶颈结构
#-----------------------------#
def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    in_channels = backend.int_shape(inputs)[-1]
    prefix = 'block_{}_'.format(block_id)

    x = inputs
    pointwise_filters = _make_divisible(int(filters * alpha), 8)
    #---------------------------------#
    #   part1 利用1x1卷积进行通道上升
    #---------------------------------#
    if block_id:
        x = Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
        x = Activation(relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    if stride == 2:
        x = ZeroPadding2D(padding=correct_pad(x, 3),
                                 name=prefix + 'pad')(x)
    
    #---------------------------------#
    #   part2 进行3x3的深度可分离卷积
    #---------------------------------#
    x = DepthwiseConv2D(kernel_size=3,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               padding='same' if stride == 1 else 'valid',
                               name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
    x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

    #-----------------------------------------------------------#   
    #   part3 利用1x1卷积进行通道的下降
    #   而且不使用relu函数，保证特征不被破坏
    #-----------------------------------------------------------#
    x = Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x

def MobileNetV2(input_shape=[224,224,3],
                alpha=1.0,
                include_top=True,
                classes=3):
    img_input = Input(shape=input_shape)

    # stem部分
    # 224,224,3 -> 112,112,32
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = ZeroPadding2D(padding=correct_pad(img_input, 3), name='Conv1_pad')(img_input)
    x = Conv2D(first_block_filters,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='valid',
                      use_bias=False,
                      name='Conv1')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
    x = Activation(relu6, name='Conv1_relu')(x)

    # 112,112,32 -> 112,112,16
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0)

    # 112,112,16 -> 56,56,24
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2)

    # 56,56,24 -> 28,28,32
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5)

    # 28,28,32 -> 14,14,64
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                            expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=9)
    # 14,14,64 -> 14,14,96
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=12)

    # 14,14,96 -> 7,7,160
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                            expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=15)
    # 7,7,160 -> 7,7,320
    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                            expansion=6, block_id=16)

    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    # 7,7,320 -> 7,7,1280
    x = Conv2D(last_block_filters, kernel_size=1, use_bias=False, name='Conv_1')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
    x = Activation(relu6, name='out_relu')(x)

    # 7,7,1280 -> 1280 -> num_classes 
    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax', use_bias=True, name='Logits')(x)

    inputs = img_input

    model = Model(inputs, x, name='mobilenetv2_%0.2f_%s' % (alpha, input_shape[0]))
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
        model = MobileNetV2(input_shape=(64, 64, 3))
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
    plt.savefig(path2+f'MobileNetV2_acc({dataset}).tif')
    plt.savefig(path2+f'MobileNetV2_acc({dataset}).png')
    plt.clf()
    plt.plot(epochs,loss,'r',label='Training loss')
    plt.plot(epochs,val_loss,'b',label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and validation loss of ResNet-50 on group data (c)')
    plt.legend(loc=2)
    plt.grid(axis="y",linewidth=1)
    plt.grid(axis="x",linewidth=1)
    plt.savefig(path2+f'MobileNetV2_loss({dataset}).tif')
    plt.savefig(path2+f'MobileNetV2_loss({dataset}).png')
    model.save(path2+f'ResNet50({dataset}).h5')

    x=np.load(path+'/'+'x_test.npy')
    y=np.load(path+'/'+'y_test.npy')
    model.evaluate(x,y)
    