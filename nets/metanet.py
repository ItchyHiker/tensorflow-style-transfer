import tensorflow as tf

from utils.layers import *

def transformer(x, weights, biases):
    with tf.variable_scope('tnet', reuse=tf.AUTO_REUSE):
        conv1 = conv(x, 'conv1', filters=32, kernel_size=9, strides=1)
        conv2 = fixed_conv(conv1, 'conv2', weights['conv2'], biases['conv2'], strides=2)
        conv3 = fixed_conv(conv2, 'conv3', weights['conv3'], biases['conv3'], strides=2)
        res1 = fixed_residual_block(conv3, 'res1', weights['res1_1'], biases['res1_1'], \
            weights['res1_2'], biases['res1_2'])
        res2 = fixed_residual_block(res1, 'res2', weights['res2_1'], biases['res2_1'], \
            weights['res2_2'], biases['res2_2'])
        res3 = fixed_residual_block(res2, 'res3', weights['res3_1'], biases['res3_1'], \
            weights['res3_2'], biases['res3_2'])
        res4 = fixed_residual_block(res3, 'res4', weights['res4_1'], biases['res4_1'], \
            weights['res4_2'], biases['res4_2'])
        res5 = fixed_residual_block(res4, 'res5', weights['res5_1'], biases['res5_1'], \
            weights['res5_2'], biases['res5_2'])
        up1 = fixed_upsample(res5, 'up1', weights['up1'], biases['up1'], strides=2)
        up2 = fixed_upsample(up1, 'up2', weights['up2'], biases['up2'], strides=2)
        conv4 = conv(up2, 'conv4', filters=3, kernel_size=9, strides=1, norm=None, act=None)

    return tf.clip_by_value(conv4, 0, 255)

def meta(vgg_out):
    conv1_2, conv2_2, conv3_3, conv4_3 = vgg_out
    conv1_2_mean, conv1_2_var = tf.nn.moments(conv1_2, axes=[1,2])
    conv2_2_mean, conv2_2_var = tf.nn.moments(conv2_2, axes=[1,2])
    conv3_3_mean, conv3_3_var = tf.nn.moments(conv3_3, axes=[1,2])
    conv4_3_mean, conv4_3_var = tf.nn.moments(conv4_3, axes=[1,2])
    concat = tf.concat([conv1_2_mean, conv1_2_var, conv2_2_mean, conv2_2_var, conv3_3_mean, \
        conv3_3_var, conv4_3_mean, conv4_3_var], axis=1)
    dense = tf.layers.dense(concat, units=1792)
    split = tf.split(dense, num_or_size_splits=14, axis=1)

    weights = {}
    biases = {}

    weights['conv2'] = tf.reshape(tf.layers.dense(split[0], units=3*3*32*64), shape=(3, 3, 32, 64))
    biases['conv2'] = tf.squeeze(tf.layers.dense(split[0], units=64))
    weights['conv3'] = tf.reshape(tf.layers.dense(split[1], units=3*3*64*128), shape=(3, 3, 64, 128))
    biases['conv3'] = tf.squeeze(tf.layers.dense(split[1], units=128))
    
    weights['res1_1'] = tf.reshape(tf.layers.dense(split[2], units=3*3*128*128), shape=[3, 3, 128, 128])
    biases['res1_1'] = tf.squeeze(tf.layers.dense(split[2], units=128))
    weights['res1_2'] = tf.reshape(tf.layers.dense(split[3], units=3*3*128*128), shape=[3, 3, 128, 128])
    biases['res1_2'] = tf.squeeze(tf.layers.dense(split[3], units=128))
    weights['res2_1'] = tf.reshape(tf.layers.dense(split[4], units=3*3*128*128), shape=[3, 3, 128, 128])
    biases['res2_1'] = tf.squeeze(tf.layers.dense(split[4], units=128))
    weights['res2_2'] = tf.reshape(tf.layers.dense(split[5], units=3*3*128*128), shape=[3, 3, 128, 128])
    biases['res2_2'] = tf.squeeze(tf.layers.dense(split[5], units=128))
    weights['res3_1'] = tf.reshape(tf.layers.dense(split[6], units=3*3*128*128), shape=[3, 3, 128, 128])
    biases['res3_1'] = tf.squeeze(tf.layers.dense(split[6], units=128))
    weights['res3_2'] = tf.reshape(tf.layers.dense(split[7], units=3*3*128*128), shape=[3, 3, 128, 128])
    biases['res3_2'] = tf.squeeze(tf.layers.dense(split[7], units=128))
    weights['res4_1'] = tf.reshape(tf.layers.dense(split[8], units=3*3*128*128), shape=[3, 3, 128, 128])
    biases['res4_1'] = tf.squeeze(tf.layers.dense(split[8], units=128))
    weights['res4_2'] = tf.reshape(tf.layers.dense(split[9], units=3*3*128*128), shape=[3, 3, 128, 128])
    biases['res4_2'] = tf.squeeze(tf.layers.dense(split[9], units=128))
    weights['res5_1'] = tf.reshape(tf.layers.dense(split[10], units=3*3*128*128), shape=[3, 3, 128, 128])
    biases['res5_1'] = tf.squeeze(tf.layers.dense(split[10], units=128))
    weights['res5_2'] = tf.reshape(tf.layers.dense(split[11], units=3*3*128*128), shape=[3, 3, 128, 128])
    biases['res5_2'] = tf.squeeze(tf.layers.dense(split[11], units=128))
    
    weights['up1'] = tf.reshape(tf.layers.dense(split[12], units=3*3*128*64), shape=(3, 3, 128, 64))
    biases['up1'] = tf.squeeze(tf.layers.dense(split[12], units=64))
    weights['up2'] = tf.reshape(tf.layers.dense(split[13], units=3*3*64*32), shape=(3, 3, 64, 32))
    biases['up2'] = tf.squeeze(tf.layers.dense(split[13], units=32))

    return weights, biases

