import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import Input, DepthwiseConv2D, Dropout, Flatten
from keras.regularizers import l2

from model_mobilenet.arcface_func import *
# from arcface_func import *
weight_decay = 1e-4


def Conv2d_BN(inputs, filters, alpha, kernels=(3, 3), strides=(1, 1),
              Linear=False, block_id=0):
    """
    卷积单元
    调整记录
    1.正则化
    2.zero padding
    :param inputs:输入
    :param filters:卷积个数
    :param alpha:alpha参数
    :param kernel:卷积核大小
    :param strides:步长
    :return:
    """
    filters = int(filters * alpha)
    # if paddings > 0:
    # inputs = ZeroPadding2D(padding=(paddings, paddings))(inputs)
    x = Conv2D(filters, kernels, padding='same',
               use_bias=False, strides=strides, name='conv1_%d' % block_id,
               kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization(axis=3, name='conv1_bn_%d' % block_id)(x)
    if not Linear:
        x = Activation('relu', name='conv1_relu_%d' % block_id)(x)  # 可试试PReLU激活函数
    return x


def Depthwise_Conv(inputs, pointwise_conv_filters, alpha, depth_multiplier=1,
                   kernels=(3, 3), strides=(1, 1), block_id=1):
    """
     调整记录，增加一些正则化
    :param inputs: 输入
    :param pointwise_conv_filters: 卷积核个数
    :param alpha: alpha擦书1,0.75,0.5等
    :param depth_multiplier:倍数参数
    :param strides:步长
    :param block_id:name参数
    :return:
    """

    pointwise_conv_filters = int(pointwise_conv_filters * alpha)  # 减少卷积核
    # if paddings > 0:
    #     inputs = ZeroPadding2D(padding=paddings)(inputs)
    # 3*3深度可分离卷积
    x = DepthwiseConv2D(kernels,
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        kernel_regularizer=l2(weight_decay),
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(axis=3, name='conv_dw_%d_bn' % block_id)(x)  # BN
    x = Activation('relu', name='conv_dw_%d_relu' % block_id)(x)  # ReLu
    # 1*1卷积 调整channels数
    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same', use_bias=False,
               strides=(1, 1),
               kernel_regularizer=l2(weight_decay),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=3, name='conv_pw_%d_bn' % block_id)(x)
    x = Activation('relu', name='conv_pw_%d_relu' % block_id)(x)
    return x


def Mobilenet_Arcface(input_shape=None, num_feature=128, classes=1000):
    alpha = 1.0
    depth_multiplier = 1
    # input = Input(shape=(28, 28, 1))
    input = Input(shape=input_shape)
    y = Input(shape=(classes,))
    x = Conv2d_BN(input, 64, alpha, strides=(2, 2), block_id=0)
    x = Depthwise_Conv(x, 64, alpha, depth_multiplier, block_id=1)
    x = Depthwise_Conv(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = Depthwise_Conv(x, 128, alpha, depth_multiplier, block_id=3)
    x = Depthwise_Conv(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = Depthwise_Conv(x, 128, alpha, depth_multiplier, block_id=5)
    x = Depthwise_Conv(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = Depthwise_Conv(x, 128, alpha, depth_multiplier, block_id=7)
    x = Depthwise_Conv(x, 128, alpha, depth_multiplier, block_id=8)
    x = Depthwise_Conv(x, 128, alpha, depth_multiplier, block_id=9)
    x = Depthwise_Conv(x, 128, alpha, depth_multiplier, block_id=10)
    x = Depthwise_Conv(x, 128, alpha, depth_multiplier, block_id=11)
    x = Depthwise_Conv(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    x = Depthwise_Conv(x, 512, alpha, depth_multiplier, strides=(1, 1), block_id=13)
    # 再增加2层
    # x = Conv2d_BN(x, 512, alpha, kernels=(1,1), strides=(1, 1), block_id=14)
    x = Conv2d_BN(x, 512, alpha, kernels=(1, 1), strides=(7, 6), Linear=True, block_id=15)
    x = Conv2d_BN(x, num_feature, alpha, kernels=(1, 1), strides=(1, 1), Linear=True, block_id=16)

    x = Flatten()(x)
    output = ArcFace(n_classes=classes)((x, y))
    return Model([input, y], output)


if __name__ == '__main__':
    model = Mobilenet_Arcface(input_shape=(112, 96, 3), num_feature=128, classes=10572)
    model.summary()
