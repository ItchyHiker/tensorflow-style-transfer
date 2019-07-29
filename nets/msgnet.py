import tensorflow as tf

from utils.layers import *
from utils.losses import *

def inspiration(x, name, style_gram):
    shape = x.shape.as_list()
    inferred_shape = tf.shape(x)

    with tf.variable_scope(name):
        b, w, h, c = shape[0] or inferred_shape[0], shape[1] or inferred_shape[1], \
            shape[2] or inferred_shape[2], shape[3]
        weight = tf.get_variable(shape=[1, c, c], name='w')
        x = tf.reshape(x, shape=[-1, w*h, c])
        x = x@tf.tile(weight@style_gram, [tf.shape(x)[0], 1, 1])
        x = tf.reshape(x, shape=[-1, w, h, c])

    return x

def transformer(x, style_gram=None):
    with tf.variable_scope('transformer', reuse=tf.AUTO_REUSE):
        conv1 = conv(x, 'conv1', filters=64, kernel_size=9, strides=1)
        conv2 = bottleneck(conv1, 'conv2', filters=128, strides=2)
        conv3 = bottleneck(conv2, 'conv3', filters=128, strides=2)
        if style_gram is None:
            return gram_matrix(conv3)

        ins = inspiration(conv3, 'inspiration', style_gram)
        res1 = bottleneck(ins, 'res1', filters=128)
        res2 = bottleneck(res1, 'res2', filters=128)
        res3 = bottleneck(res2, 'res3', filters=128)
        res4 = bottleneck(res3, 'res4', filters=128)
        res5 = bottleneck(res4, 'res5', filters=128)

        up1 = upbottleneck(res5, 'up1', filters=64, strides=2)
        up2 =  upbottleneck(up1, 'up2', filters=32, strides=2)

        conv4 = bottleneck_conv(up2, 'conv4', filters=3, kernel_size=9, strides=1)

    return tf.clip_by_value(conv4, 0, 255)

