
.. automodule:: astroNN.nn.callbacks
.. automodule:: astroNN.nn.utilities
.. automodule:: astroNN.nn

Callbacks and Utilities
============================

A callback is a set of functions under ``astroNN.nn.callbacks`` and ``astroNN.nn.utilities`` modules to be applied at given stages of the training procedure.
astroNN provides some customized callbacks which built on tensorflow.keras. You can just treat astroNN customized callbacks as conventional Keras callbacks.

astroNN also contains some handy utilities for data processing

Virtual CSVLogger (Callback)
-------------------------------

.. autoclass:: astroNN.nn.callbacks.VirutalCSVLogger
    :members: savefile

`VirutalCSVLogger` is basically Keras's CSVLogger without Python 2 support and won't write the file to disk until
`savefile()` method is called after the training where Keras's CSVLogger will write to disk immediately.


`VirutalCSVLogger` can be imported by

.. code-block:: python
    :linenos:
    
    from astroNN.nn.callbacks import VirutalCSVLogger

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python
    :linenos:
    
    def keras_model():
        # Your keras_model define here
        return model

    # Create a Virtual_CSVLogger instance first
    csvlogger = VirutalCSVLogger()

    # Default filename is training_history.csv
    # You have to set filename first before passing to Keras
    csvlogger.filename = 'training_history.csv'

    model = keras_model()
    model.compile(....)

    model.fit(...,callbacks=[csvlogger])

    # Save the file to current directory
    csvlogger.savefile()

    # OR to save the file to other directory
    csvlogger.savefile(folder_name='some_folder')

Raising Error on Nan (Callback)
-----------------------------------

.. autoclass:: astroNN.nn.callbacks.ErrorOnNaN

`ErrorOnNaN` is basically Keras's TerminateOnNaN but will raise `ValueError` on Nan, its useful for python unittest to
make sure you can catch the error and know something is wrong.

.. automodule:: astroNN.nn.utilities.normalizer


Normalizer (Utility)
-----------------------

astroNN `Normalizer` is called when `train()` method is called and involved `pre_training_checklist_master()` method
defined in `NeuralNetMaster` Class. `Normalizer` will not normalize data/labels equal to ``magicnumber`` defined in configuration file.
So that astroNN loss function can recognize those missing/bad data.

`Normalizer` consists of a few modes that you can, but the mode will minus mean and divide standard derivation to the data.


.. math::

    \text{Normalized Data} = \frac{\text{Data} - \text{Mean}}{\text{Standard Derivation}} \text{for Data} \neq \text{Magic Number}

1. `Mode 0` means normalizing data with mean=0 and standard derivation=1 (same as doing nothing)

.. code-block:: python
    :linenos:
    
    # If we have some data
    data = np.array([[1,2,3], [9,8,7]])

    # THe normalized data, mean std will as follow by this mode
    norm_data = array([[1,2,3], [9,8,7]])
    # the mean and standard derivation used to do the normalization
    mean = [0.]
    std = [1.]

2. `Mode 1` means normalizing data with a single mean and a single standard derivation of the data

.. code-block:: python
    :linenos:
    
    # If we have some data
    data = np.array([[1,2,3], [9,8,7]])

    # THe normalized data, mean std will as follow by this mode
    norm_data = array([[-1.28653504, -0.96490128, -0.64326752], [ 1.28653504,  0.96490128,  0.64326752]])
    # the mean and standard derivation used to do the normalization
    mean = [5.0]
    std = [3.11]

3. `Mode 2` means normalizing data with pixelwise means and pixelwise standard derivations of the data

.. code-block:: python
    :linenos:
    
    # If we have some data
    data = np.array([[1,2,3], [9,8,7]])

    # THe normalized data, mean std will as follow by this mode
    norm_data = array([[-4., -3., -2.], [ 4.,  3.,  2.]])
    # the mean and standard derivation used to do the normalization
    mean = [5., 5., 5.]
    std = [4., 3., 2.]

4. `Mode 3` means normalizing data with featurewise mean and standard derivation=1 the data (only centered the data), it is useful for normalizing spectra

.. code-block:: python
    :linenos:
    
    # If we have some data
    data = array([[1,2,3], [9,8,7]])

    # THe normalized data, mean std will as follow by this mode
    norm_data = array([[-1., -1., -1.], [ 1.,  1.,  1.]])
    # the mean and standard derivation used to do the normalization
    mean = [5., 5., 5.]
    std = [1.]

5. `Mode 3s` means normalizing data with featurewise mean and standard derivation=1 the data (only centered the data), then apply sigmoid for normalization or sigmoid inverse for denormalization. It is useful for normalizing spectra for Variational Autoencoder with Negative Log Likelihood objective.

6. `Mode 255` means normalizing data with mean=127.5 and standard derivation=127.5, this mode is designed to normalize 8bit images

.. code-block:: python
    :linenos:
    
    # If we have some data
    data = np.array([[255,125,100], [99,87,250]])

    # THe normalized data, mean std will as follow by this mode
    norm_data = array([[ 1. , -0.01960784, -0.21568627], [-0.22352941, -0.31764706,  0.96078431]])
    # the mean and standard derivation used to do the normalization
    mean = [127.5]
    std = [127.5]

You can set the mode from a astroNN neural net instance before called `train()` method by

.. code-block:: python
    :linenos:
    
    # To set the normalization mode for input and labels
    astronn_neuralnet.input_norm_mode = ...
    astronn_neuralnet.labels_norm_mode = ...

You can use `Normalizer()` independently to take advantage of this function won't touch data equal ``magicnumber``.
`Normalizer()` always return you the normalized data, the mean and standard derivation used to do the normalization

.. code-block:: python
    :linenos:
    
    from astroNN.nn.utilities.normalizer import Normalizer
    import numpy as np

    # Make some data up
    data = np.array([[1.,2.,3.], [9.,8.,7.]])

    # Setup a normalizer instance with a mode, lets say mode 1
    normer = Normalizer(mode=1)

    # Use the instance method normalize to normalize the data
    norm_data = normer.normalize(data)

    print(norm_data)
    >>> array([[-1.28653504, -0.96490128, -0.64326752], [ 1.28653504,  0.96490128,  0.64326752]])
    print(normer.mean_labels)
    >>> 5.0
    print(normer.std_labels)
    >>> 3.1091263510296048

    # You can use the same instance (with same mean and std and mode) to demoralize data
    denorm_data = normer.denormalize(data)

    print(denorm_data)
    >>> array([[1.,2.,3.], [9.,8.,7.]])


NumPy Implementation of Tensorflow function - **astroNN.nn.numpy**
------------------------------------------------------------------------

astroNN has some handy numpy implementation of a number of tensorflow functions. The list of available functions are

.. automodule:: astroNN.nn.numpy
    :members:
