import tensorflow as tf

def VGG16(x, weights):
    mean = tf.constant([123.68, 116.779, 103.99], dtype=tf.float32, shape=[1, 1, 1, 3], name='vgg_mean')
    x = tf.subtract(x, mean)

    with tf.variable_scope('vgg16', reuse=tf.AUTO_REUSE):
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.get_variable(initializer=tf.constant(weights["conv1_1_W"]),
                trainable=False, name='conv1_1_W')
            biases = tf.get_variable(initializer=tf.constant(weights['conv1_1_b']),
                trainable=False, name='conv1_1_b')
            conv1_1 = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
            conv1_1 = tf.nn.bias_add(conv1_1, biases)
            conv1_1 = tf.nn.relu(conv1_1, name=scope)
        
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.get_variable(initializer=tf.constant(weights["conv1_2_W"]),
                trainable=False, name='conv1_2_W')
            biases = tf.get_variable(initializer=tf.constant(weights['conv1_2_b']),
                trainable=False, name='conv1_2_b')
            conv1_2 = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            conv1_2 = tf.nn.bias_add(conv1_2, biases)
            conv1_2 = tf.nn.relu(conv1_2, name=scope)
       
        pool1 = tf.nn.avg_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
            padding='SAME', name='pool1')
        
        
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.get_variable(initializer=tf.constant(weights["conv2_1_W"]),
                trainable=False, name='conv2_1_W')
            biases = tf.get_variable(initializer=tf.constant(weights['conv2_1_b']),
                trainable=False, name='conv2_1_b')
            conv2_1 = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            conv2_1 = tf.nn.bias_add(conv2_1, biases)
            conv2_1 = tf.nn.relu(conv2_1, name=scope)
        
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.get_variable(initializer=tf.constant(weights["conv2_2_W"]),
                trainable=False, name='conv2_2_W')
            biases = tf.get_variable(initializer=tf.constant(weights['conv2_2_b']),
                trainable=False, name='conv2_2_b')
            conv2_2 = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            conv2_2 = tf.nn.bias_add(conv2_2, biases)
            conv2_2 = tf.nn.relu(conv2_2, name=scope)
       
        pool2 = tf.nn.avg_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
            padding='SAME', name='pool2')
        
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.get_variable(initializer=tf.constant(weights["conv3_1_W"]),
                trainable=False, name='conv3_1_W')
            biases = tf.get_variable(initializer=tf.constant(weights['conv3_1_b']),
                trainable=False, name='conv3_1_b')
            conv3_1 = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            conv3_1 = tf.nn.bias_add(conv3_1, biases)
            conv3_1 = tf.nn.relu(conv3_1, name=scope)
        
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.get_variable(initializer=tf.constant(weights["conv3_2_W"]),
                trainable=False, name='conv3_2_W')
            biases = tf.get_variable(initializer=tf.constant(weights['conv3_2_b']),
                trainable=False, name='conv3_2_b')
            conv3_2 = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            conv3_2 = tf.nn.bias_add(conv3_2, biases)
            conv3_2 = tf.nn.relu(conv3_2, name=scope)
       
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.get_variable(initializer=tf.constant(weights["conv3_3_W"]),
                trainable=False, name='conv3_3_W')
            biases = tf.get_variable(initializer=tf.constant(weights['conv3_3_b']),
                trainable=False, name='conv3_3_b')
            conv3_3 = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            conv3_3 = tf.nn.bias_add(conv3_3, biases)
            conv3_3 = tf.nn.relu(conv3_3, name=scope)
        
        
        pool3 = tf.nn.avg_pool(conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
            padding='SAME', name='pool3')
        
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.get_variable(initializer=tf.constant(weights["conv4_1_W"]),
                trainable=False, name='conv4_1_W')
            biases = tf.get_variable(initializer=tf.constant(weights['conv4_1_b']),
                trainable=False, name='conv4_1_b')
            conv4_1 = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            conv4_1 = tf.nn.bias_add(conv4_1, biases)
            conv4_1 = tf.nn.relu(conv4_1, name=scope)
        
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.get_variable(initializer=tf.constant(weights["conv4_2_W"]),
                trainable=False, name='conv4_2_W')
            biases = tf.get_variable(initializer=tf.constant(weights['conv4_2_b']),
                trainable=False, name='conv4_2_b')
            conv4_2 = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            conv4_2 = tf.nn.bias_add(conv4_2, biases)
            conv4_2 = tf.nn.relu(conv4_2, name=scope)

        with tf.name_scope('conv4_3') as scope:
            kernel = tf.get_variable(initializer=tf.constant(weights["conv4_3_W"]),
                trainable=False, name='conv4_3_W')
            biases = tf.get_variable(initializer=tf.constant(weights['conv4_3_b']),
                trainable=False, name='conv4_3_b')
            conv4_3 = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            conv4_3 = tf.nn.bias_add(conv4_3, biases)
            conv4_3 = tf.nn.relu(conv4_3, name=scope)
        
    return conv1_2, conv2_2, conv3_3, conv4_3

