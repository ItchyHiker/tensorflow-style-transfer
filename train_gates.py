import time
import argparse

import numpy as np
from PIL import Image
import tensorflow as tf

import config
from utils.losses import *
from vgg16 import VGG16

def train(style_img_path, content_img_path, num_iters, learning_rate, style_size, 
    content_size, log_dir, style_loss_weights, content_loss_weights, reg_loss_weight, 
    vgg_weights_path, log_iter=100):
    style_img = tf.keras.preprocessing.image.load_img(style_img_path, target_size=(style_size, style_size))
    style_img = tf.keras.preprocessing.image.img_to_array(style_img)
    
    content_img = tf.keras.preprocessing.image.load_img(content_img_path, target_size=(content_size, content_size))
    content_img = tf.keras.preprocessing.image.img_to_array(content_img)
    
    vgg_weights = np.load(vgg_weights_path)

    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    training_graph = tf.Graph()
    
    with training_graph.as_default() as g, tf.Session(config=config) as sess:
        s = sess.run(tf.expand_dims(style_img, axis=0))
        c = sess.run(tf.expand_dims(content_img, axis=0))

        s_placeholder = tf.placeholder(name='style', dtype=tf.float32, shape=[1, style_size, style_size, 3])
        c_placeholder = tf.placeholder(name='content', dtype=tf.float32, shape=[1, content_size, content_size, 3])
        
        transferred = tf.clip_by_value(tf.Variable(initial_value=c, dtype=tf.float32), 0, 255)
        target_style_features = VGG16(s_placeholder, vgg_weights)
        target_content_features = VGG16(c_placeholder, vgg_weights)
        transferred_features = VGG16(transferred, vgg_weights)
        
        loss = loss_func(target_style_features, target_content_features, transferred_features,
            transferred, style_loss_weights, content_loss_weights, reg_loss_weight)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        loss_summary = tf.summary.scalar('loss', tf.squeeze(loss))
        summary = tf.summary.FileWriter(graph=g, logdir=log_dir)

        sess.run(tf.global_variables_initializer())

        start = time.time()

        for i in range(num_iters):
            _, cur_loss, cur_loss_summary = sess.run([optimizer, loss, loss_summary], feed_dict={s_placeholder: s, c_placeholder: c})
            if (i+1) % log_iter == 0:
                print("Iteration: {0}, loss: {1}".format(i+1, cur_loss))

            summary.add_summary(cur_loss_summary, i+1)
            summary.flush()

        end = time.time()

        result = sess.run(tf.squeeze(transferred))

    print("Finished {num_iters} iterations in {time} seconds.".format(num_iters=num_iters, time=end-start))
                    
    Image.fromarray(result.astype('uint8')).save('imgs/result/gates_result.jpg')


def main(args):
    if args.train:
        train(config.STYLE_IMG,
            config.CONTENT_IMG,
            num_iters=1000,
            learning_rate=config.learning_rate,
            style_size=256,
            content_size=512,
            log_dir='./logs/gates',
            style_loss_weights=config.style_loss_weights,
            content_loss_weights=config.content_loss_weights,
            reg_loss_weight=config.reg_loss_weight,
            vgg_weights_path='/home/ubuntu/weights/vgg16_weights.npz')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False)
    args = parser.parse_args()
    main(args)
