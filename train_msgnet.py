import os
import time
import argparse
import shutil

import numpy as np
from PIL import Image
import tensorflow as tf

import config
from utils.losses import *
from vgg16 import VGG16
from nets.msgnet import transformer

tf.set_random_seed(2)
np.random.seed(2)

def train(style_img_path, content_img_path, num_epochs, learning_rate, style_size, 
    content_size, log_dir, style_loss_weights, content_loss_weights, reg_loss_weight, 
    vgg_weights_path, ckpt_dir, log_iter=100, sample_iter=100, content_batch_size=4):
    
    
    iterator = tf.keras.preprocessing.image.DirectoryIterator
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    content_iter = iterator(directory=content_img_path, batch_size=content_batch_size, \
        target_size=(content_size, content_size), image_data_generator=datagen, shuffle=True, seed=2)
    style_iter = iterator(directory=style_img_path, batch_size=1, target_size=\
        (style_size, style_size), image_data_generator=datagen, seed=2)

    total_iteration = num_epochs * content_iter.n // content_batch_size
    
    vgg_weights = np.load(vgg_weights_path)

    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    training_graph = tf.Graph()
    
    with training_graph.as_default() as g, tf.Session(config=config) as sess:

        s_placeholder = tf.placeholder(name='style', dtype=tf.float32, shape=[1, style_size, style_size, 3])
        c_placeholder = tf.placeholder(name='content', dtype=tf.float32, shape=[content_batch_size, content_size, content_size, 3])
        
        target_style_features = VGG16(s_placeholder, vgg_weights)
        target_content_features = VGG16(c_placeholder, vgg_weights)
        style_gram = transformer(s_placeholder)

        transferred = transformer(c_placeholder, style_gram)
        transferred_features = VGG16(transferred, vgg_weights)
        
        loss = loss_func(target_style_features, target_content_features, transferred_features,
            transferred, style_loss_weights, content_loss_weights, reg_loss_weight)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        
        loss_summary = tf.summary.scalar('loss', tf.squeeze(loss))
        style_summary = tf.summary.image('style', s_placeholder)
        content_summary = tf.summary.image('content', c_placeholder)
        transferred_summary = tf.summary.image('transferred', transferred)
        image_summary = tf.summary.merge([style_summary, content_summary, transferred_summary])

        summary = tf.summary.FileWriter(graph=g, logdir=log_dir)


        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables())

        start = time.time()
        
        for epoch in range(num_epochs):
            i = 0
            content_iter.reset()
            style_iter.reset()
            for c, _ in content_iter:
                if i+1 == (content_iter.n // content_batch_size) :
                    break
                
                try:
                    s,  _ = style_iter.next()
                except StopIteration:
                    style_iter.reset()
                    s, _ = style_iter.next()

                _, cur_loss, cur_loss_summary, cur_image_summary \
                = sess.run([optimizer, loss, loss_summary, image_summary], feed_dict={s_placeholder: s, c_placeholder: c})
            
                if (i+1) % log_iter == 0:
                    print("Iteration: {0}, loss: {1}".format(epoch*content_iter.n // 4 + i+1, cur_loss))
                
                summary.add_summary(cur_loss_summary, epoch*content_iter.n // 4 + i+1)
                
                if (i+1) % sample_iter == 0:
                    summary.add_summary(cur_image_summary, epoch*content_iter.n // 4 + i+1)

                summary.flush()
                i += 1          
            save_path = os.path.join(ckpt_dir, 'ckpt')
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = saver.save(sess, save_path, write_meta_graph=False, global_step=epoch*content_iter.n // 4 + i+1)
            print("Checkpoint saved as: {ckpt_path}".format(ckpt_path=ckpt_path))
        
        end = time.time()

    print("Finished {num_iters} iterations in {time} seconds.".format(num_iters=total_iteration, time=end-start))
                    
def export_saved_model(style_size, ckpt_dir, export_dir):
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    tf.reset_default_graph()
    eval_graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    exporter = tf.saved_model.builder.SavedModelBuilder(export_dir)
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)

    with eval_graph.as_default() as g, tf.Session(config=config, graph=eval_graph) as sess:
        s_placeholder = tf.placeholder(name='style', dtype=tf.float32, shape=[1, style_size, style_size, 3])
        inputs = tf.placeholder(name='inputs', dtype=tf.float32, shape=[None, None, None, 3])
        
        style_gram_out = tf.identity(transformer(s_placeholder), name='style_gram_out')
        style_gram_in = tf.placeholder(name='style_gram_in', dtype=tf.float32, shape=style_gram_out.shape)
        outputs = tf.identity(transformer(inputs, style_gram_in), name='outputs')
        saver = tf.train.Saver()
        saver.restore(sess, latest_ckpt)

        exporter.add_meta_graph_and_variables(
            sess,
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf.saved_model.signature_def_utils.predict_signature_def(inputs={'inputs': inputs,
                                                                                 'style': s_placeholder, 
                                                                                 'style_gram_in':style_gram_in},
                                                                         outputs={'outputs': outputs, 
                                                                                  'style_gram_out': style_gram_out})

            })
        
        exporter.save()

def eval_with_saved_model(saved_model_dir, style_img, content_img, style_size):
    style_img = tf.keras.preprocessing.image.img_to_array(img=tf.keras.preprocessing.image.load_img(style_img, target_size=(style_size, style_size)))
    content_img = tf.keras.preprocessing.image.img_to_array(img=tf.keras.preprocessing.image.load_img(content_img))

    tf.reset_default_graph()
    eval_graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with eval_graph.as_default() as g, tf.Session(config=config, graph=eval_graph) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
        style = g.get_tensor_by_name('style:0')
        inputs = g.get_tensor_by_name('inputs:0')
        outputs = g.get_tensor_by_name('outputs:0')
        style_gram_out = g.get_tensor_by_name('style_gram_out:0')
        style_gram_in = g.get_tensor_by_name('style_gram_in:0')

        c, s = sess.run([tf.expand_dims(content_img, axis=0), tf.expand_dims(style_img, axis=0)])
        style_gram = sess.run(style_gram_out, feed_dict={style:s})
        start = time.time()
        result = sess.run(tf.squeeze(outputs), feed_dict={inputs:c, style_gram_in:style_gram})
        end = time.time()
        print('Inference time: {time} seconds'.format(time=end-start))
        Image.fromarray(result.astype('uint8')).save('imgs/result/msgnet_result.jpg')

def main(args):
    if args.train:
        train(config.STYLE_IMG_PATH,
            config.CONTENT_IMG_PATH,
            num_epochs=config.epochs,
            learning_rate=config.learning_rate,
            style_size=config.style_img_size,
            content_size=config.content_img_size,
            log_dir=config.log_dir,
            style_loss_weights=config.style_loss_weights,
            content_loss_weights=config.content_loss_weights,
            reg_loss_weight=config.reg_loss_weight,
            vgg_weights_path='/home/ubuntu/weights/vgg16_weights.npz', 
            ckpt_dir=config.ckpt_dir)
    else:
        export_saved_model(config.style_img_size, './ckpt/msgnet', './saved_model/msgnet')
        eval_with_saved_model('./saved_model/msgnet', config.STYLE_IMG, config.CONTENT_IMG, config.style_img_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False)
    args = parser.parse_args()
    main(args)
