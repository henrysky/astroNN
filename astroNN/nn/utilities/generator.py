import threading
from abc import ABC, abstractmethod

import numpy as np
from astroNN.config import keras_import_manager

keras = keras_import_manager()
Sequence = keras.utils.data_utils.Sequence

# class ThreadSafeIter(object):
#     """
#     Takes an iterator/generator and makes it thread-safe by
#     serializing call to the `next` method of given iterator/generator.
#     """
#
#     def __init__(self, it):
#         self.it = it
#         self.lock = threading.Lock()
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         with self.lock:
#             return self.it.__next__()
#
#
# def threadsafe_generator(f):
#     """
#     A decorator that takes a generator function and makes it thread-safe.
#     """
#
#     def g(*a, **kw):
#         return ThreadSafeIter(f(*a, **kw))
#
#     return g


class GeneratorMaster(Sequence):
    """Top-level class for a generator"""

    def __init__(self, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _get_exploration_order(self, idx_list):
        """
        :param idx_list:
        :return:
        """
        # shuffle (if applicable) and find exploration order
        indexes = np.copy(idx_list)
        if self.shuffle is True:
            np.random.shuffle(indexes)

        return indexes

    def sparsify(self, y):
        """Returns labels in binary NumPy array"""
        # n_classes =  # Enter number of classes
        # return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
        #                  for i in range(y.shape[0])])
        pass

    def input_d_checking(self, inputs, idx_list_temp):
        if inputs.ndim == 2:
            x = np.empty((self.batch_size, inputs.shape[1], 1))
            # Generate data
            x[:, :, 0] = inputs[idx_list_temp]

        elif inputs.ndim == 3:
            x = np.empty((self.batch_size, inputs.shape[1], inputs.shape[2], 1))
            # Generate data
            x[:, :, :, 0] = inputs[idx_list_temp]

        elif inputs.ndim == 4:
            x = np.empty((self.batch_size, inputs.shape[1], inputs.shape[2], inputs.shape[3]))
            # Generate data
            x[:, :, :, :] = inputs[idx_list_temp]
        else:
            raise ValueError(f"Unsupported data dimension, your data has {inputs.ndim} dimension")

        return x

    @abstractmethod
    def _data_generation(self, *args):
        pass

    @abstractmethod
    def generate(self, *args):
        pass
