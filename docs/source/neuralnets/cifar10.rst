Cifar10 with astroNN
=======================

Here is a Cifar10 example using astroNN

.. code:: python

    from keras.datasets import cifar10
    from keras.utils import np_utils
    import numpy as np

    from astroNN.models import Cifar10CNN

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    y_train = y_train.astype(np.float32)
    x_train = x_train.astype(np.float32)

    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    net = Cifar10CNN()
    net.max_epochs = 10
    net.train(x_train, y_train)

.. code:: python

    # Load the folder back
    from astroNN.models import load_folder

    # Replace with correct name
    cnn = load_folder('astroNN_0114_run001')
    prediction = cnn.test(x_test)
    print(prediction)
