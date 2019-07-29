import os
import time
import argparse

import tensorflow as tf
from PIL import Image

import config


def main(saved_model_dir, content_img):
    content_img = tf.keras.preprocessing.image.img_to_array(img=tf.keras.preprocessing.image.load_img(content_img))
    tf.reset_default_graph()
    eval_graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with eval_graph.as_default() as g, tf.Session(config=config, graph=eval_graph) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
        inputs = g.get_tensor_by_name('inputs:0')
        outputs = g.get_tensor_by_name('outputs:0')
        c = sess.run(tf.expand_dims(content_img, axis=0))
        start = time.time()
        result = sess.run(tf.squeeze(outputs), feed_dict={inputs:c})
        end = time.time()
        print("Inferene  time: {time} seconds.".format(time=end-start))

        Image.fromarray(result.astype('uint8')).save('result.jpg')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model_dir', type=str)
    args = parser.parse_args()

    main(args.saved_model_dir, config.CONTENT_IMG)


