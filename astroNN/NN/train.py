# ---------------------------------------------------------#
#   astroNN.NN.train: download gaia files
# ---------------------------------------------------------#

import tensorflow as tf


def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))