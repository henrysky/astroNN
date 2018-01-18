import threading
from abc import ABC, abstractmethod

import numpy as np


class ThreadSafeIter(object):
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return ThreadSafeIter(f(*a, **kw))

    return g


class GeneratorMaster(ABC):
    """Top-level class for a generator"""

    def __init__(self, batch_size, shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _get_exploration_order(self, list_IDs):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle is True:
            np.random.shuffle(indexes)

        return indexes

    def sparsify(self, y):
        'Returns labels in binary NumPy array'
        # n_classes =  # Enter number of classes
        # return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
        #                  for i in range(y.shape[0])])
        pass

    def input_d_checking(self, inputs, list_IDs_temp):
        if inputs.ndim == 2:
            X = np.empty((self.batch_size, inputs.shape[1], 1))
            # Generate data
            X[:, :, 0] = inputs[list_IDs_temp]

        elif inputs.ndim == 3:
            X = np.empty((self.batch_size, inputs.shape[1], inputs.shape[2], 1))
            # Generate data
            X[:, :, :, 0] = inputs[list_IDs_temp]

        elif inputs.ndim == 4:
            X = np.empty((self.batch_size, inputs.shape[1], inputs.shape[2], inputs.shape[3]))
            # Generate data
            X[:, :, :, :] = inputs[list_IDs_temp]
        else:
            raise TypeError

        return X

    @abstractmethod
    def _data_generation(self, *args):
        pass

    @abstractmethod
    @threadsafe_generator
    def generate(self, *args):
        pass