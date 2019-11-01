import os
import shutil
import sys
import unittest
from importlib import import_module

import numpy as np
import tensorflow.keras as tfk

import astroNN
from astroNN.config import config_path
from astroNN.models import Cifar10CNN, Galaxy10CNN, MNIST_BCNN
from astroNN.models import load_folder
from astroNN.nn.callbacks import ErrorOnNaN

mnist = tfk.datasets.mnist
utils = tfk.utils


class Models_TestCase(unittest.TestCase):
    def test_mnist(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        y_train = utils.to_categorical(y_train, 10)

        # To convert to desirable type
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.astype(np.float32)

        # create model instance
        mnist_test = Cifar10CNN()
        mnist_test.max_epochs = 1
        mnist_test.callbacks = ErrorOnNaN()

        mnist_test.train(x_train[:200], y_train[:200])
        output_shape = mnist_test.output_shape
        mnist_test.test(x_test[:200])
        mnist_test.evaluate(x_train[:100], y_train[:100])

        # create model instance for binary classification
        mnist_test = Cifar10CNN()
        mnist_test.max_epochs = 1
        mnist_test.task = 'binary_classification'

        mnist_test.train(x_train[200:400], y_train[200:400].astype(bool))
        prediction = mnist_test.test(x_test[200:400])

        mnist_test.save('mnist_test')
        mnist_reloaded = load_folder("mnist_test")
        prediction_loaded = mnist_reloaded.test(x_test[200:400])
        mnist_reloaded.jacobian_old(x_test[:2])
        eval_result = mnist_reloaded.evaluate(x_test[200:400], y_train[200:400].astype(bool))

        # Cifar10_CNN without dropout is deterministic
        np.testing.assert_array_equal(prediction, prediction_loaded)

        # test verbose metrics
        mnist_reloaded.metrics = ['accuracy']
        mnist_reloaded.compile()
        mnist_test.save('mnist_test_accuracy')
        mnist_reloaded_again = load_folder("mnist_test_accuracy")
        eval_result_again = mnist_reloaded_again.evaluate(x_test[200:400], y_train[200:400])
        # assert saving again wont affect the model
        self.assertAlmostEqual(eval_result_again['loss'], eval_result['loss'], places=3)


class Models_TestCase2(unittest.TestCase):
    def test_color_images(self):
        # test colored 8bit images
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.random.randint(0, 255, size=(200, 28, 28, 3))
        x_test = np.random.randint(0, 255, size=(100, 28, 28, 3))
        y_train = y_train[:200]
        y_train = utils.to_categorical(y_train, 10)
        # To convert to desirable type

        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.astype(np.float32)

        # create model instance
        mnist_test = Cifar10CNN()
        mnist_test.max_epochs = 1
        mnist_test.callbacks = ErrorOnNaN()

        mnist_test.train(x_train, y_train[:200])
        mnist_test.test(x_test[:200])

        # create model instance for binary classification
        mnist_test = Galaxy10CNN()
        mnist_test.max_epochs = 1

        mnist_test.train(x_train[:200], y_train[:200])
        prediction = mnist_test.test(x_test[:200])

        mnist_test.save('cifar10_test')
        mnist_reloaded = load_folder("cifar10_test")
        prediction_loaded = mnist_reloaded.test(x_test[:200])
        mnist_reloaded.jacobian(x_test[:2], mean_output=True, mc_num=2)
        # mnist_reloaded.hessian_diag(x_test[:10], mean_output=True, mc_num=2)

        # Cifar10_CNN is deterministic
        np.testing.assert_array_equal(prediction, prediction_loaded)


class Models_TestCase3(unittest.TestCase):
    def test_bayesian_mnist(self):
        import pylab as plt
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        y_train = utils.to_categorical(y_train, 10)
        # To convert to desirable type
        y_train = y_train.astype(np.float32)
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)

        # Create a astroNN neural network instance and set the basic parameter
        net = MNIST_BCNN()
        net.task = 'classification'
        net.callbacks = ErrorOnNaN()
        net.max_epochs = 1  # Just use 1 epochs for quick result

        # Train the neural network
        net.train(x_train[:200], y_train[:200])
        net.save('mnist_bcnn_test')
        net.plot_dense_stats()
        plt.close()  # Travis-CI memory error??
        net.evaluate(x_train[:10], y_train[:10])

        net_reloaded = load_folder("mnist_bcnn_test")
        net_reloaded.mc_num = 3  # prevent memory issue on Tavis CI
        prediction_loaded = net_reloaded.test(x_test[:200])

        net_reloaded.folder_name = None  # set to None so it can be saved
        net_reloaded.save()

        load_folder(net_reloaded.folder_name)  # ignore pycharm warning, its not None


class Models_TestCase4(unittest.TestCase):
    def setUp(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        y_train = utils.to_categorical(y_train, 10)
        self.y_train = y_train.astype(np.float32)
        self.x_train = x_train.astype(np.float32)
        self.x_test = x_test.astype(np.float32)

    def test_bayesian_binary_mnist(self):
        # Create a astroNN neural network instance and set the basic parameter
        net = MNIST_BCNN()
        net.task = 'binary_classification'
        net.callbacks = ErrorOnNaN()
        net.max_epochs = 1  # Just use 1 epochs for quick result

        # Train the neural network
        net.train(self.x_train[:200], self.y_train[:200])

        net.save('mnist_binary_bcnn_test')
        net_reloaded = load_folder("mnist_binary_bcnn_test")
        net_reloaded.mc_num = 3  # prevent memory issue on Tavis CI
        prediction_loaded = net_reloaded.test(self.x_test[:200])


class Models_TestCase5(unittest.TestCase):
    def test_custom_model(self):
        # get config path and remove it so we can copy and paste the new one
        astroNN_config_path = config_path()
        if os.path.exists(astroNN_config_path):
            os.remove(astroNN_config_path)

        # copy and paste the new one from test suite
        test_config_path = os.path.join(os.path.dirname(astroNN.__path__[0]), 'tests', 'config.ini')
        shutil.copy(test_config_path, astroNN_config_path)

        # copy and paste the custom model from test suite to travis user space
        test_modelsource_path = os.path.join(os.path.dirname(astroNN.__path__[0]), 'tests', 'custom_model',
                                             'custom_models.py')
        shutil.copy(test_modelsource_path, os.path.join('/home/travis/build/henrysky', 'custom_models.py'))

        head, tail = os.path.split(test_modelsource_path)

        sys.path.insert(0, head)
        CustomModel_Test = getattr(import_module(tail.strip('.py')), str('CustomModel_Test'))

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        y_train = utils.to_categorical(y_train, 10)

        # To convert to desirable type
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.astype(np.float32)

        # disable due to travis error
        # create model instance
        # custom_model = CustomModel_Test()
        # custom_model.max_epochs = 1
        #
        # custom_model.train(x_train[:200], y_train[:200])
        #
        # prediction = custom_model.test(x_test[:200])
        # custom_model.save('custom_model_testing_folder')
        #
        # custom_model_loaded = load_folder("custom_model_testing_folder")
        # prediction_loaded = custom_model_loaded.test(x_test[:200])
        # # CustomModel_Test is deterministic
        # np.testing.assert_array_equal(prediction, prediction_loaded)


class Models_TestCase6(unittest.TestCase):
    def test_nomodel(self):
        nomodel = Galaxy10CNN()
        self.assertRaises(AttributeError, nomodel.summary)
        self.assertRaises(AttributeError, nomodel.save)
        self.assertRaises(AttributeError, nomodel.get_weights)
        self.assertRaises(AttributeError, nomodel.test, np.zeros(100))

    def test_load_flawed_fodler(self):
        from astroNN.config import astroNN_CACHE_DIR
        self.assertRaises(FileNotFoundError, load_folder, astroNN_CACHE_DIR)
        self.assertRaises(IOError, load_folder, 'i_am_not_a_fodler')


if __name__ == '__main__':
    unittest.main()
