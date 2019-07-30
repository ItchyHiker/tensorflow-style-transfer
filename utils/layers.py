import tensorflow as tf

def instance_norm(x, name, epsilon=1e-5):
    with tf.variable_scope(name):
        gamma = tf.get_variable(initializer=tf.ones([x.shape[-1]]), name='gamma')
        beta = tf.get_variable(initializer=tf.ones([x.shape[-1]]), name='beta')
        mean, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name='norm')
    return x

def conv(x, name, filters, kernel_size, strides, norm=instance_norm, act=tf.nn.relu):
    padding = kernel_size // 2
    with tf.variable_scope(name):
        x = tf.pad(x, paddings=[[0,0], [padding, padding], [padding, padding], [0, 0]], mode='REFLECT')
        x = tf.layers.conv2d(x, filters, kernel_size=kernel_size, strides=strides, name='conv')
        if norm is not None:
            x = norm(x, name='norm')
        if act is not None:
            x = act(x, name='act')
    return x

def bottleneck_conv(x, name, filters, kernel_size, strides, norm=instance_norm, act=tf.nn.relu):
    padding = kernel_size // 2
    with tf.variable_scope(name):
        if norm is not None:
            x = norm(x, name='norm')
        if act is not None:
            x = act(x, name='act')
        x = tf.pad(x, paddings=[[0,0], [padding, padding], [padding, padding], [0, 0]], mode='REFLECT')
        x = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, name='conv')
    return x

def bottleneck(x, name, filters, strides=1):
    with tf.variable_scope(name):
        residual = conv(x, 'residual', filters, kernel_size=1, strides=2, norm=None, act=None) if strides > 1 else x
        x = bottleneck_conv(x, 'conv1', filters//4, kernel_size=1, strides=1)
        x = bottleneck_conv(x, 'conv2', filters//4, kernel_size=3, strides=strides)
        x = bottleneck_conv(x, 'conv3', filters, kernel_size=1, strides=1)
    return x + residual

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
    x = tf.image.resize_images(x, size=[w*strides, h*strides])
    x = conv(x, name, filters, kernel_size, strides=1)
    return x

def upbottleneck_conv(x, name, filters, kernel_size, strides, norm=instance_norm, act=tf.nn.relu):
    shape = x.shape.as_list()
    inferred_shape = tf.shape(x)
    w, h = shape[1] or inferred_shape[1], shape[2] or inferred_shape[2]
    x = tf.image.resize_images(x, size=[w*strides, h*strides])
    x = bottleneck_conv(x, name, filters, kernel_size, strides=1, norm=norm, act=act)
    return x

def upbottleneck(x, name, filters, strides=1):
    with tf.variable_scope(name):
        residual = upbottleneck_conv(x, 'residual', filters, kernel_size=1, strides=2, norm=None, act=None) \
        if strides > 1 else x
        x = upbottleneck_conv(x, 'conv1', filters//4, kernel_size=1, strides=1)
        x = upbottleneck_conv(x, 'conv2', filters//4, kernel_size=3, strides=strides)
        x = upbottleneck_conv(x, 'conv3', filters, kernel_size=1, strides=1)
    return x + residual

def fixed_conv(x, name, conv_w, conv_b, strides, norm=instance_norm, act=tf.nn.relu):
    padding = conv_w.shape[1] // 2
    with tf.variable_scope(name):
        x = tf.pad(x, paddings=[[0,0],[padding, padding], [padding, padding], [0,0]], mode='REFLECT')
        x = tf.nn.conv2d(x, conv_w, [1, strides, strides, 1], 'VALID', name='conv')
        x = tf.nn.bias_add(x, conv_b)
        if norm is not None:
            x = norm(x, name='norm')
        if act is not None:
            x = act(x, name='act')
    return x

def fixed_residual_block(x, name, conv1_w, conv1_b, conv2_w, conv2_b):
    with tf.variable_scope(name):
        residual = x
        x = fixed_conv(x, 'conv1', conv1_w, conv1_b, strides=1)
        x = fixed_conv(x, 'conv2', conv2_w, conv2_b, strides=1, act=None)
    return x + residual

def fixed_upsample(x, name, conv_w, conv_b, strides):
    shape = x.shape.as_list()
    inferred_shape = tf.shape(x)
    w, h = shape[1] or inferred_shape[1], shape[2] or inferred_shape[2]
    x = tf.image.resize_images(x, size=[w*strides, h*strides])
    x = fixed_conv(x, name, conv_w, conv_b, strides=1)
    return x


