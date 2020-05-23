from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from model import common
import tensorflow as tf

def darknet53(input_data, trainable):
    '''
    主干网络
    :param input_data:
    :param trainable:
    :return:
    '''
    with tf.compat.v1.variable_scope('darknet'):
        #fileters_shape A Tensor. Must have the same type as input. A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
        input_data = common.convolution(input_data, fileters_shape = (3, 3, 3, 32), trainable=trainable,name = 'conv0')
        input_data = common.convolution(input_data, fileters_shape=(3, 3, 32,  64),
                                          trainable=trainable, name='conv1', down_sample=True)
        for i in range(1):
            input_data = common.residual_block(input_data,  64,  32, 64, trainable=trainable, name='residual%d' %(i+0))
        input_data = common.convolution(input_data, fileters_shape = (3, 3, 64, 128), trainable = trainable, name='conv2')
        for i in range(2):
            input_data = common.residual_block(input_data, 128, 64, 128, trainable = trainable, name = 'residual%d'%(i + 1))
        input_data = common.convolution(input_data, fileters_shape=(3, 3, 128, 256), trainable=trainable, name='conv3')
        for i in range(8):
            route_1 = input_data
            input_data = common.residual_block(input_data, 256, 128, 256, trainable = trainable, name = 'residual%d'%(i + 3))
        input_data = common.convolution(input_data, fileters_shape=(3, 3, 256, 512), trainable=trainable, name='conv4')
        for i in range(8):
            route_2 = input_data
            input_data = common.residual_block(input_data, 512, 256, 512, trainable = trainable, name = 'residual%d'%(i + 11))
        input_data = common.convolution(input_data, fileters_shape=(3, 3, 512, 1024), trainable=trainable, name='conv5')
        for i in range(4):
            input_data = common.residual_block(input_data, 1024, 512, 1024, trainable = trainable, name = 'residual%d'%(i + 19))

    return route_1, route_2, input_data #3个不同尺度

