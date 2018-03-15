Introduction to astroNN Neural Nets
=======================================================

Available astroNN Neural Net Classes
--------------------------------------

All astroNN Neural Nets are inherited from some child classes which inherited NeuralNetMaster, NeuralNetMaster also
relies relies on two major component, `Normalizer` and `GeneratorMaster`

::

    Normalizer (astroNN.nn.utilities.normalizer.Normalizer)

    GeneratorMaster (astroNN.nn.utilities.generator.GeneratorMaster)
    ├── CNNDataGenerator
    ├── Bayesian_DataGenerator
    └── CVAE_DataGenerator

    NeuralNetMaster (astroNN.models.NeuralNetMaster.NeuralNetMaster)
    ├── CNNBase
    │   ├── ApogeeCNN
    │   ├── StarNet2017
    │   └── Cifar10CNN
    ├── BayesianCNNBase
    │   ├── MNIST_BCNN  # For authors testing only
    │   └── ApogeeBCNN
    ├── ConvVAEBase
    │   └── APGOEECVAE  # For authors testing only
    └── CGANBase
        └── GalaxyGAN2017  # For authors testing only

NeuralNetMaster Class
--------------------------------------

NeuralNetMaster is the top level abstract class for all astroNN sub neural network classes. NeuralNetMaster define the
structure of how an astroNN neural network class should look like.

NeuralNetMaster consists of a pre-training checking (check input and labels shape, cpu/gpu check and create astroNN
folder for every run.

---------------------------------------------------------------
When `train()` is called from an astroNN neural net instance
---------------------------------------------------------------

When `train()` is called, the method will call `pre_training_checklist_child()` defined in the corresponding child class
and call `pre_training_checklist_master()` defined in `NeuralNetMaster`. `pre_training_checklist_master()` basically responsible
to do basic data checking, create an astroNN folder for this run and save hyperparameters.

After `pre_training_checklist_master()` has finished, `pre_training_checklist_child()` will run its checklist, including
normalizing data, compile model and setup the data generator which will yield data to the neural net during training.

Normalizer
---------------

astroNN `Normalizer` is called when `train()` method is called and involved `pre_training_checklist_master()` method
defined in `NeuralNetMaster` Class. `Normalizer` will not normalize data/labels equal to ``magicnumber`` defined in configuration file.
So that astroNN loss function can recognize those missing/bad data.

`Normalizer` consists of a few modes that you can, but the mode will minus mean and divide standard derivation to the data.


.. math::

    \text{Normalized Data} = \frac{\text{Data} - \text{Mean}}{\text{Standard Derivation}} \text{for Data} \neq \text{Magic Number}

1. `Mode 0` means normalizing data with mean=0 and standard derivation=1 (same as doing nothing)

.. code-block:: python

    # If we have some data
    data = np.array([[1,2,3], [9,8,7]])

    # THe normalized data, mean std will as follow by this mode
    norm_data = array([[1,2,3], [9,8,7]])
    # the mean and standard derivation used to do the normalization
    mean = [0.]
    std = [1.]

2. `Mode 1` means normalizing data with a single mean and a single standard derivation of the data

.. code-block:: python

    # If we have some data
    data = np.array([[1,2,3], [9,8,7]])

    # THe normalized data, mean std will as follow by this mode
    norm_data = array([[-1.28653504, -0.96490128, -0.64326752], [ 1.28653504,  0.96490128,  0.64326752]])
    # the mean and standard derivation used to do the normalization
    mean = [5.0]
    std = [3.11]

3. `Mode 2` means normalizing data with pixelwise means and pixelwise standard derivations of the data

.. code-block:: python

    # If we have some data
    data = np.array([[1,2,3], [9,8,7]])

    # THe normalized data, mean std will as follow by this mode
    norm_data = array([[-4., -3., -2.], [ 4.,  3.,  2.]])
    # the mean and standard derivation used to do the normalization
    mean = [5., 5., 5.]
    std = [4., 3., 2.]

4. `Mode 3` means normalizing data with featurewise mean and standard derivation=1 the data (only centered the data), it is useful for normalizing spectra

.. code-block:: python

    # If we have some data
    data = array([[1,2,3], [9,8,7]])

    # THe normalized data, mean std will as follow by this mode
    norm_data = array([[-1., -1., -1.], [ 1.,  1.,  1.]])
    # the mean and standard derivation used to do the normalization
    mean = [5., 5., 5.]
    std = [1.]

5. `Mode 255` means normalizing data with mean=127.5 and standard derivation=127.5, this mode is designed to normalize 8bit images

.. code-block:: python

    # If we have some data
    data = np.array([[255,125,100], [99,87,250]])

    # THe normalized data, mean std will as follow by this mode
    norm_data = array([[ 1. , -0.01960784, -0.21568627], [-0.22352941, -0.31764706,  0.96078431]])
    # the mean and standard derivation used to do the normalization
    mean = [127.5]
    std = [127.5]

You can set the mode from a astroNN neural net instance before called `train()` method by

.. code-block:: python

    # To set the normalization mode for input and labels
    astronn_neuralnet.input_norm_mode = ...
    astronn_neuralnet.labels_norm_mode = ...

You can use `Normalizer()` independently to take advantage of this function won't touch data equal ``magicnumber``.
`Normalizer()` always return you the normalized data, the mean and standard derivation used to do the normalization

.. code-block:: python

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

CNNBase Class
--------------------------------------
