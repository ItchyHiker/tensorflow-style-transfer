import tensorflow as tf

def gram_matrix(x):
    b, w, h, c = x.shape.as_list()
    x = tf.reshape(x, [b, w*h, c])
    return tf.matmul(x, x, transpose_a=True) / (c*w*h)

def loss_func(target_style_features, target_content_features, transferred_features, 
    transferred, style_loss_weights, content_loss_weights, reg_loss_weight):
    content_loss = 0.
    style_loss = 0.
    reg_loss = 0.
    
    for i in range(len(transferred_features)):
        if content_loss_weights[i] != 0:
            content_loss += content_loss_weights[i] * \
            tf.nn.l2_loss(target_content_features[i] - transferred_features[i])
        if style_loss_weights[i] != 0:
            gram_target = gram_matrix(target_style_features[i])
            gram_transferred = gram_matrix(transferred_features[i])
            style_loss += style_loss_weights[i] * \
            tf.nn.l2_loss(gram_target - gram_transferred)
    if reg_loss_weight != 0:
        reg_loss = reg_loss_weight * tf.image.total_variation(transferred)
    
    return content_loss + style_loss + reg_loss
    
