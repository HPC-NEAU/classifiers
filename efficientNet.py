from __future__ import absolute_import, division, print_function

import math
import os
from copy import deepcopy

import numpy as np
import tensorflow as tf
from keras import backend, layers
from keras.applications import imagenet_utils
from keras_applications.imagenet_utils import (_obtain_input_shape, decode_predictions)  # 版本问题可修改
from keras.models import Model
from keras.preprocessing import image
from keras.utils.data_utils import get_file

#-------------------------------------------------#
#   用于下载模型的默认参数
#-------------------------------------------------#
BASE_WEIGHTS_PATH = ('https://github.com/Callidior/keras-applications/releases/download/efficientnet/')
WEIGHTS_HASHES = {
    'b0': ('e9e877068bd0af75e0a36691e03c072c',
           '345255ed8048c2f22c793070a9c1a130'),
    'b1': ('8f83b9aecab222a9a2480219843049a1',
           'b20160ab7b79b7a92897fcb33d52cc61'),
    'b2': ('b6185fdcd190285d516936c09dceeaa4',
           'c6e46333e8cddfa702f4d8b8b6340d70'),
    'b3': ('b2db0f8aac7c553657abb2cb46dcbfbb',
           'e0cf8654fad9d3625190e30d70d0c17d'),
    'b4': ('ab314d28135fe552e2f9312b31da6926',
           'b46702e4754d2022d62897e0618edc7b'),
    'b5': ('8d60b903aff50b09c6acf8eaba098e09',
           '0a839ac36e46552a881f2975aaab442f'),
    'b6': ('a967457886eac4f5ab44139bdd827920',
           '375a35c17ef70d46f9c664b03b4437f2'),
    'b7': ('e964fd6e26e9a4c144bcb811f2a10f20',
           'd55674cc46b805f4382d18bc08ed43c1')
}

#-------------------------------------------------#
#   用于计算padding的大小
#-------------------------------------------------#
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

#-------------------------------------------------#
#   一共七个大结构块，每个大结构块都有特定的参数
#-------------------------------------------------#
DEFAULT_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

#--------------------------------#
#   两个Kernel的初始化器
#--------------------------------#
CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'normal'
    }
}
DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

#-------------------------------------------------#
#   Swish激活函数
#-------------------------------------------------#
def get_swish():
    def swish(x):
        return x * backend.sigmoid(x)
    return swish

def block(inputs, activation_fn=get_swish, drop_rate=0., name='',
          filters_in=32, filters_out=16, kernel_size=3, strides=1,
          expand_ratio=1, se_ratio=0., id_skip=True):

    filters = filters_in * expand_ratio
    #-------------------------------------------------#
    #   利用Inverted residuals
    #   part1 利用1x1卷积进行通道数上升
    #-------------------------------------------------#
    if expand_ratio != 1:
        x = layers.Conv2D(filters, 1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=name + 'expand_conv')(inputs)
        x = layers.BatchNormalization(name=name + 'expand_bn')(x)
        x = layers.Activation(activation_fn, name=name + 'expand_activation')(x)
    else:
        x = inputs

    #------------------------------------------------------#
    #   如果步长为2x2的话，利用深度可分离卷积进行高宽压缩
    #   part2 利用3x3卷积对每一个channel进行卷积
    #------------------------------------------------------#
    if strides == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(x, kernel_size),
                                 name=name + 'dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
    x = layers.DepthwiseConv2D(kernel_size,
                               strides=strides,
                               padding=conv_pad,
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=name + 'dwconv')(x)
    x = layers.BatchNormalization(name=name + 'bn')(x)
    x = layers.Activation(activation_fn, name=name + 'activation')(x)

    #------------------------------------------------------#
    #   完成深度可分离卷积后
    #   对深度可分离卷积的结果施加注意力机制
    #------------------------------------------------------#
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
        se = layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
        #------------------------------------------------------#
        #   通道先压缩后上升，最后利用sigmoid将值固定到0-1之间
        #------------------------------------------------------#
        se = layers.Conv2D(filters_se, 1,
                           padding='same',
                           activation=activation_fn,
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + 'se_reduce')(se)
        se = layers.Conv2D(filters, 1,
                           padding='same',
                           activation='sigmoid',
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + 'se_expand')(se)
        x = layers.multiply([x, se], name=name + 'se_excite')

    #------------------------------------------------------#
    #   part3 利用1x1卷积进行通道下降
    #------------------------------------------------------#
    x = layers.Conv2D(filters_out, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=name + 'project_conv')(x)
    x = layers.BatchNormalization(name=name + 'project_bn')(x)

    #------------------------------------------------------#
    #   part4 如果满足残差条件，那么就增加残差边
    #------------------------------------------------------#
    if (id_skip is True and strides == 1 and filters_in == filters_out):
        if drop_rate > 0:
            x = layers.Dropout(drop_rate,
                               noise_shape=(None, 1, 1, 1),
                               name=name + 'drop')(x)
        x = layers.add([x, inputs], name=name + 'add')

    return x


def EfficientNet(width_coefficient,
                 depth_coefficient,
                 default_size,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 activation_fn=tf.nn.swish,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 model_name='efficientnet',
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=3,
                 **kwargs):

    img_input = layers.Input(tensor=input_tensor, shape=input_shape)

    #-------------------------------------------------#
    #   该函数的目的是保证filter的大小可以被8整除
    #-------------------------------------------------#
    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    #-------------------------------------------------#
    #   计算模块的重复次数
    #-------------------------------------------------#
    def round_repeats(repeats):
        return int(math.ceil(depth_coefficient * repeats))

    #-------------------------------------------------#
    #   创建stem部分
    #-------------------------------------------------#
    x = img_input
    x = layers.ZeroPadding2D(padding=correct_pad(x, 3), name='stem_conv_pad')(x)
    x = layers.Conv2D(round_filters(32), 3,
                      strides=2,
                      padding='valid',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(x)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation(activation_fn, name='stem_activation')(x)

    blocks_args = deepcopy(blocks_args)
    #-------------------------------------------------#
    #   计算总的efficient_block的数量
    #-------------------------------------------------#
    b = 0
    blocks = float(sum(args['repeats'] for args in blocks_args))
    #------------------------------------------------------------------------------#
    #   对结构块参数进行循环、一共进行7个大的结构块。
    #   每个大结构块下会重复小的efficient_block
    #------------------------------------------------------------------------------#
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        #-------------------------------------------------#
        #   对使用到的参数进行更新
        #-------------------------------------------------#
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            x = block(x, activation_fn, drop_connect_rate * b / blocks, name='block{}{}_'.format(i + 1, chr(j + 97)), **args)
            b += 1
    
    #-------------------------------------------------#
    #   1x1卷积调整通道数
    #-------------------------------------------------#
    x = layers.Conv2D(round_filters(1280), 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='top_conv')(x)
    x = layers.BatchNormalization(name='top_bn')(x)
    x = layers.Activation(activation_fn, name='top_activation')(x)

    #-------------------------------------------------#
    #   利用GlobalAveragePooling2D代替全连接层
    #-------------------------------------------------#
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name='top_dropout')(x)

    x = layers.Dense(classes, activation='softmax', kernel_initializer=DENSE_KERNEL_INITIALIZER, name='probs')(x)

    inputs = img_input
    model = Model(inputs, x, name=model_name)

    #-------------------------------------------------#
    #   载入权值
    #-------------------------------------------------#
    if weights == 'imagenet':
        file_suff = '_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
        file_hash = WEIGHTS_HASHES[model_name[-2:]][0]
        file_name = model_name + file_suff
        weights_path = get_file(file_name,BASE_WEIGHTS_PATH + file_name,
                                    cache_subdir='models',
                                    file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model

def EfficientNetB0(weights=None,
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=3,
                   **kwargs):
    return EfficientNet(1.0, 1.0, 224, 0.2,
                        model_name='efficientnet-b0',
                        weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)

def EfficientNetB1(weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.0, 1.1, 240, 0.2,
                        model_name='efficientnet-b1',
                        weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)

def EfficientNetB2(weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.1, 1.2, 260, 0.3,
                        model_name='efficientnet-b2',
                        weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)

def EfficientNetB3(weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.2, 1.4, 300, 0.3,
                        model_name='efficientnet-b3',
                        weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB4(weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.4, 1.8, 380, 0.4,
                        model_name='efficientnet-b4',
                        weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)

def EfficientNetB5(weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.6, 2.2, 456, 0.4,
                        model_name='efficientnet-b5',
                        weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)

def EfficientNetB6(weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.8, 2.6, 528, 0.5,
                        model_name='efficientnet-b6',
                        weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)

def EfficientNetB7(weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(2.0, 3.1, 600, 0.5,
                        model_name='efficientnet-b7',
                        weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)

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
        model = EfficientNetB0(input_shape=[64,64,3])
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
    plt.savefig(path2+f'EfficientNet_acc({dataset}).tif')
    plt.savefig(path2+f'EfficientNet_acc({dataset}).png')
    plt.clf()
    plt.plot(epochs,loss,'r',label='Training loss')
    plt.plot(epochs,val_loss,'b',label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and validation loss of ResNet-50 on group data (c)')
    plt.legend(loc=2)
    plt.grid(axis="y",linewidth=1)
    plt.grid(axis="x",linewidth=1)
    plt.savefig(path2+f'EfficientNet_loss({dataset}).tif')
    plt.savefig(path2+f'EfficientNet_loss({dataset}).png')
    model.save(path2+f'ResNet50({dataset}).h5')

    x=np.load(path+'/'+'x_test.npy')
    y=np.load(path+'/'+'y_test.npy')
    model.evaluate(x,y)
    