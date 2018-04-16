import unittest

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from astroNN.models import Cifar10CNN, Galaxy10CNN
from astroNN.models import load_folder


class Models_TestCase(unittest.TestCase):
    def test_mnist(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        y_train = np_utils.to_categorical(y_train, 10)

        # To convert to desirable type
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.astype(np.float32)

        # create model instance
        mnist_test = Cifar10CNN()
        mnist_test.max_epochs = 1

        mnist_test.train(x_train[:1000], y_train[:1000])
        mnist_test.test(x_test[:1000])

        # create model instance for binary classification
        mnist_test = Cifar10CNN()
        mnist_test.max_epochs = 1
        mnist_test.task = 'binary_classification'

        mnist_test.train(x_train[:1000], y_train[:1000])
        prediction = mnist_test.test(x_test[:1000])

        mnist_test.save('mnist_test')
        mnist_reloaded = load_folder("mnist_test")
        prediction_loaded = mnist_reloaded.test(x_test[:1000])
        mnist_reloaded.jacobian_old(x_test[:10])

        # Cifar10_CNN is deterministic
        np.testing.assert_array_equal(prediction, prediction_loaded)

    def test_color_images(self):
        # test colored 8bit images
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.random.randint(0, 255, size=(1000, 28, 28, 3))
        x_test = np.random.randint(0, 255, size=(100, 28, 28, 3))
        y_train = y_train[:1000]
        y_train = np_utils.to_categorical(y_train, 10)
        # To convert to desirable type

        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.astype(np.float32)

        # create model instance
        mnist_test = Cifar10CNN()
        mnist_test.max_epochs = 1

        mnist_test.train(x_train, y_train[:1000])
        mnist_test.test(x_test[:1000])

        # create model instance for binary classification
        mnist_test = Galaxy10CNN()
        mnist_test.max_epochs = 1

        mnist_test.train(x_train[:1000], y_train[:1000])
        prediction = mnist_test.test(x_test[:1000])

        mnist_test.save('cifar10_test')
        mnist_reloaded = load_folder("cifar10_test")
        prediction_loaded = mnist_reloaded.test(x_test[:1000])
        mnist_reloaded.jacobian(x_test[:10], mean_output=True, mc_num=2)

        # Cifar10_CNN is deterministic
        np.testing.assert_array_equal(prediction, prediction_loaded)

    def test_bayesian_mnist(self):
        from astroNN.models import MNIST_BCNN
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        y_train = np_utils.to_categorical(y_train, 10)
        y_train = y_train.astype(np.float32)
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)

        # Create a astroNN neural network instance and set the basic parameter
        net = MNIST_BCNN()
        net.task = 'classification'
        net.max_epochs = 1  # Just use 5 epochs for quick result

        # Trian the nerual network
        net.train(x_train[:1000], y_train[:1000])

        net.save('mnist_bcnn_test')
        net_reloaded = load_folder("mnist_bcnn_test")
        net_reloaded.mc_num = 3  # prevent memory issue on Tavis CI
        prediction_loaded = net_reloaded.test(x_test[:1000])

    def test_load_flawed_fodler(self):
        from astroNN.config import astroNN_CACHE_DIR
        self.assertRaises(FileNotFoundError, load_folder, astroNN_CACHE_DIR)
        self.assertRaises(IOError, load_folder, 'i_am_not_a_fodler')

    def test_custom_model(self):
        import shutil
        import os
        import astroNN
        from astroNN.config import config_path

        test_config_path = os.path.join(os.path.dirname(astroNN.__path__[0]), 'tests', 'config.ini')
        astroNN_config_path = config_path()
        if os.path.exists(astroNN_config_path):
            os.remove(astroNN_config_path)
        shutil.copy(test_config_path, astroNN_config_path)

        test_modelsource_path = os.path.join(os.path.dirname(astroNN.__path__[0]), 'tests', 'custom_model',
                                             'custom_models.py')
        shutil.copy(test_modelsource_path, os.path.join('/home/travis/build/henrysky', 'custom_models.py'))

        import sys
        from importlib import import_module
        head, tail = os.path.split(test_modelsource_path)

        sys.path.insert(0, head)
        CustomModel_Test = getattr(import_module(tail.strip('.py')), str('CustomModel_Test'))

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        y_train = np_utils.to_categorical(y_train, 10)

        # To convert to desirable type
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.astype(np.float32)

        # create model instance
        custom_model = CustomModel_Test()
        custom_model.max_epochs = 1

        custom_model.train(x_train[:1000], y_train[:1000])
        prediction = custom_model.test(x_test[:1000])

        custom_model.save('custom_model_testing_folder')

        custom_model_loaded = load_folder("custom_model_testing_folder")
        prediction_loaded = custom_model_loaded.test(x_test[:1000])
        # CustomModel_Test is deterministic
        np.testing.assert_array_equal(prediction, prediction_loaded)


if __name__ == '__main__':
    unittest.main()
