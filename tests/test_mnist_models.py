import unittest
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

from astroNN.models import Cifar10_CNN


class MNIST_TestCase(unittest.TestCase):
    def test_mnist(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        y_train = np_utils.to_categorical(y_train, 10)

        # To convert to desirable type
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.astype(np.float32)

        # create model instance
        mnist_test = Cifar10_CNN()
        mnist_test.max_epochs = 1

        mnist_test.train(x_train[:1000], y_train[:1000])
        mnist_test.test(x_test[:1000])

        # create model instance for binary classification
        mnist_test = Cifar10_CNN()
        mnist_test.max_epochs = 1
        mnist_test.task = 'binary_classification'

        mnist_test.train(x_train[:1000], y_train[:1000])
        mnist_test.test(x_test[:1000])



if __name__ == '__main__':
    unittest.main()
