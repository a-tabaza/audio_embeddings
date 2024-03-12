from __future__ import print_function
import numpy as np
import resampy  # pylint: disable=import-error
import tensorflow.compat.v1 as tf
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

import audiofile
import audresample

import sys

checkpoint_path = 'vggish_model.ckpt'
pca_params_path = 'vggish_pca_params.npz'

target_rate = 16000

def embed_audio(path, pooling="mean", postprocess=False):
    signal, sampling_rate = audiofile.read(path)
    if sampling_rate != target_rate:
        signal = audresample.resample(signal, sampling_rate, target_rate)
    if signal.shape[0] == 2:
        mixed = audresample.remix(
        signal,
        mixdown=True,
        )

    input_batch = vggish_input.waveform_to_examples(mixed.reshape(-1, 1).squeeze(), target_rate)

    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim()
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)
        [embedding_batch] = sess.run([embedding_tensor],
                                    feed_dict={features_tensor: input_batch})
    if pooling == "mean":
        if postprocess:
            pproc = vggish_postprocess.Postprocessor(pca_params_path)
            postprocessed_batch = pproc.postprocess(embedding_batch)
            return tf.reduce_mean(postprocessed_batch, 0).numpy()
        return tf.linalg.normalize(tf.reduce_mean(embedding_batch, 0))[0].numpy()
    if pooling == "sum":
        if postprocess:
            pproc = vggish_postprocess.Postprocessor(pca_params_path)
            postprocessed_batch = pproc.postprocess(embedding_batch)
            return tf.reduce_sum(postprocessed_batch, 0).numpy()
        return tf.linalg.normalize(tf.reduce_sum(embedding_batch, 0))[0].numpy()
    

if __name__ == "__main__":
    pprocess = bool(sys.argv[3])
    print(embed_audio(sys.argv[1], sys.argv[2], pprocess))