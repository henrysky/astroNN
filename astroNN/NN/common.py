# ---------------------------------------------------------#
#   astroNN.NN.common: common functions shared by modules
# ---------------------------------------------------------#

import tensorflow as tf


def get_session(gpu_ram_fraction=None):
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
