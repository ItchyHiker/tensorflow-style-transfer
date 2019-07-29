import tensorflow as tf

def instance_norm(x, name, epsilon=1e-5):
    with tf.variable_scope(name):
        gamma = tf.get_variable(initializer=tf.ones([x.shape[-1]]), name='gamma')
        beta = tf.get_variable(initializer=tf.ones([x.shape[-1]]), name='beta')
        mean, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name='norm')

    return x

def conv(x, name, filters, kernel_size, strides, norm=instance_norm, act=tf.nn.relu):
    padding = kenrel_size // 2
    with tf.variable_scope(name):
        x = tf.pad(x, paddings=[[0,0], [padding, padding], [padding, padding], [0, 0]], mode='REFLECT')
        x = tf.layers.conv2d(x, filters, kernel_size=kernel_size, strides=strides, name='conv')
        if norm is not None:
            x = norm(x, name='norm')

        if act is not None:
            x = act(x, name='act')

def residual_block(x, name, filters, kernel_size):
    with tf.variable_scope(name):
        residual = x
        x = conv(x, 'conv1', filters, kernel_size, strides=1)
        x = conv(x, 'conv2', filters, kernel_size, strides=1, act=None)

    return x + residual

def upsample(x, name, filters, kernel_size, strides):
    shape = x.shape.as_list()
    inferred_shape = tf.shape(x)

    w, h = shape[1] or inferred_shape[1], shape[2] or inferred_shape[2]
    x = tf.image.resize_image(x, size=[w*strides, h*strides])
    x = conv(x, name, filters, kernel_size, strides=1)
    return x

   

