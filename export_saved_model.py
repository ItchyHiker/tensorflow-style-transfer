import os
import shutil
import argparse

import tensorflow as tf

from nets.perceptual import transformer

def main(ckpt_dir, export_dir):
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    tf.reset_default_graph()
    eval_graph = tf.Graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    exporter = tf.saved_model.builder.SavedModelBuilder(export_dir)
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)

    with eval_graph.as_default() as g, tf.Session(config=config, graph=eval_graph) as sess:
        inputs = tf.placeholder(name='inputs', dtype=tf.float32, shape=[None, None, None, 3])
        outputs = tf.identity(transformer(inputs), name='outputs')

        saver = tf.train.Saver()
        saver.restore(sess, latest_ckpt)

        exporter.add_meta_graph_and_variables(
            sess,
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf.saved_model.signature_def_utils.predict_signature_def(inputs={"inputs":inputs},
                                                                         outputs={"outputs":outputs})
            })

        exporter.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--export_dir', type=str)

    args = parser.parse_args()

    main(args.ckpt_dir, args.export_dir)


