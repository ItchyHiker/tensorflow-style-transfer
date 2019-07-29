import tensorflow as tf

from utils.layers import *

def transformer(x):
    with tf.variable_scope('transformer', reuse=tf.AUTO_REUSE):
        conv1 = conv(x, 'conv1', filters=32, kernel_size=9, strides=1)
        conv2 = conv(conv1, 'conv2', filters=64, kernel_size=3, strides=2)
        conv3 = conv(conv2, 'conv3', filters=128, kernel_size=3, strides=2)
        res1 = residual_block(conv3, 'res1', filters=128, kernel_size=3)
        res2 = residual_block(res1, 'res2', filters=128, kernel_size=3)
        res3 = residual_block(res2, 'res3', filters=128, kernel_size=3)
        res4 = residual_block(res3, 'res4', filters=128, kernel_size=3)
        res5 = residual_block(res4, 'res5', filters=128, kernel_size=3)
        up1 = upsample(res5, 'up1', filters=64, kernel_size=3, strides=2)
        up2 = upsample(up1, 'up2', filters=32, kernel_size=3, strides=2)
        conv4 = conv(up2, 'conv4', filters=3, kernel_size=9, strides=1, norm=None, act=None)

    return tf.clip_by_value(conv4, 0, 255.)


def transformer_moblie():
    pass

