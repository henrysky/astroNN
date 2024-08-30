import os
import shutil
import sys
from importlib import import_module

import keras
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest
from astroNN.config import astroNN_CACHE_DIR, config_path
from astroNN.models import MNIST_BCNN, Cifar10CNN, Galaxy10CNN, load_folder
from astroNN.nn.callbacks import ErrorOnNaN


def test_mnist(mnist_data):
    # create model instance
    x_train, y_train, x_test, y_test = mnist_data

    mnist_test = Cifar10CNN()
    mnist_test.max_epochs = 1
    mnist_test.callbacks = ErrorOnNaN()

    mnist_test.fit(x_train, y_train)
    pred = mnist_test.predict(x_test)
    test_num = y_test.shape[0]
    assert (
        np.sum(np.argmax(pred, axis=1) == np.argmax(y_test, axis=1))
    ) / test_num > 0.9  # assert accurancy
    mnist_test.evaluate(x_test, y_test)

    # create model instance for binary classification
    mnist_test = Cifar10CNN()
    mnist_test.max_epochs = 2
    mnist_test.task = "binary_classification"

    mnist_test.fit(x_train, y_train.astype(bool))
    prediction = mnist_test.predict(x_test)
    assert (
        np.sum(np.argmax(prediction, axis=1) == np.argmax(y_test, axis=1))
    ) / test_num > 0.9  # assert accuracy
    mnist_test.save("mnist_test")
    mnist_reloaded = load_folder("mnist_test")
    prediction_loaded = mnist_reloaded.predict(x_test)
    eval_result = mnist_reloaded.evaluate(x_test, y_test)

    # Cifar10_CNN without dropout is deterministic
    np.testing.assert_array_equal(prediction, prediction_loaded)

    # test verbose metrics
    mnist_reloaded.metrics = ["accuracy"]
    mnist_reloaded.compile()
    mnist_test.save("mnist_test_accuracy")
    mnist_reloaded_again = load_folder("mnist_test_accuracy")
    # test with astype boolean deliberately
    eval_result_again = mnist_reloaded_again.evaluate(x_test, y_test.astype(bool))
    # assert saving again wont affect the model
    npt.assert_almost_equal(eval_result_again["loss"], eval_result["loss"], decimal=3)


def test_color_images(mnist_data):
    # create model instance
    x_train, y_train, x_test, y_test = mnist_data
    x_train_color = np.stack([x_train, x_train, x_train], axis=-1)
    x_test_color = np.stack([x_test, x_test, x_test], axis=-1)

    mnist_test = Cifar10CNN()
    mnist_test.max_epochs = 1
    mnist_test.callbacks = ErrorOnNaN()

    mnist_test.fit(x_train_color, y_train)
    pred = mnist_test.predict(x_test_color)
    test_num = y_test.shape[0]
    assert (
        np.sum(np.argmax(pred, axis=1) == np.argmax(y_test, axis=1))
    ) / test_num > 0.9  # assert accuracy

    # create model instance for binary classification
    mnist_test = Galaxy10CNN()
    mnist_test.max_epochs = 1
    mnist_test.mc_num = 3

    mnist_test.fit(x_train_color[:200], y_train[:200])
    prediction = mnist_test.predict(x_test_color[:200])

    mnist_test.save("cifar10_test")
    mnist_reloaded = load_folder("cifar10_test")
    prediction_loaded = mnist_reloaded.predict(x_test_color[:200])
    mnist_reloaded.jacobian(x_test_color[:2], mean_output=True, mc_num=2)
    mnist_reloaded.hessian(x_test_color[:2], mean_output=True, mc_num=2)

    # Cifar10_CNN is deterministic
    np.testing.assert_array_equal(prediction, prediction_loaded)


def test_bayesian_mnist(mnist_data):
    # Create a astroNN neural network instance and set the basic parameter
    x_train, y_train, x_test, y_test = mnist_data
    net = MNIST_BCNN()
    net.task = "classification"
    net.callbacks = ErrorOnNaN()
    net.max_epochs = 1

    # Train the neural network
    net.fit(x_train, y_train)
    net.save("mnist_bcnn_test")
    net.plot_dense_stats()
    plt.close()  # Travis-CI memory error??
    net.evaluate(x_test, y_test)

    pred, pred_err = net.predict(x_test)
    test_num = y_test.shape[0]
    assert (
        np.sum(pred == np.argmax(y_test, axis=1))
    ) / test_num > 0.9  # assert accuracy

    net_reloaded = load_folder("mnist_bcnn_test")
    net_reloaded.mc_num = 3  # prevent memory issue on Tavis CI
    prediction_loaded = net_reloaded.predict(x_test[:200])

    net_reloaded.folder_name = None  # set to None so it can be saved
    net_reloaded.save()

    load_folder(net_reloaded.folder_name)  # ignore pycharm warning, its not None


def test_bayesian_binary_mnist(mnist_data):
    # Create a astroNN neural network instance and set the basic parameter
    x_train, y_train, x_test, y_test = mnist_data
    net = MNIST_BCNN()
    net.task = "binary_classification"
    net.callbacks = ErrorOnNaN()
    net.max_epochs = 1
    net.fit(x_train, y_train)
    pred, pred_err = net.predict(x_test)
    test_num = y_test.shape[0]

    net.save("mnist_binary_bcnn_test")
    net_reloaded = load_folder("mnist_binary_bcnn_test")
    net_reloaded.mc_num = 3
    prediction_loaded, prediction_loaded_err = net_reloaded.predict(x_test)

    # assert (np.sum(np.argmax(pred, axis=1) == np.argmax(y_test, axis=1))) / test_num > 0.9  # assert accuracy
    # assert (np.sum(np.argmax(prediction_loaded, axis=1) == np.argmax(y_test, axis=1))) / test_num > 0.9  # assert accuracy


def test_custom_model():
    # get config path and remove it so we can copy and paste the new one
    astroNN_config_path = config_path()
    if os.path.exists(astroNN_config_path):
        os.remove(astroNN_config_path)

    # copy and paste the new one from test suite
    test_config_path = "./tests/config.ini"
    shutil.copy(test_config_path, astroNN_config_path)
    test_modelsource_path = "./tests/custom_model/custom_models.py"
    shutil.copy(test_modelsource_path, os.path.join(os.getcwd(), "custom_models.py"))

    head, tail = os.path.split(test_modelsource_path)

    sys.path.insert(0, head)
    CustomModel_Test = getattr(
        import_module(tail.strip(".py")), str("CustomModel_Test")
    )


def test_nomodel():
    nomodel = Galaxy10CNN()
    with pytest.raises(AttributeError):
        nomodel.summary()
    with pytest.raises(AttributeError):
        nomodel.save()
    with pytest.raises(AttributeError):
        nomodel.get_weights()
    with pytest.raises(AttributeError):
        nomodel.predict(np.zeros(100))


def test_load_flawed_fodler():
    with pytest.raises(FileNotFoundError):
        load_folder(astroNN_CACHE_DIR)
    with pytest.raises(IOError):
        load_folder("i_am_not_a_fodler")
