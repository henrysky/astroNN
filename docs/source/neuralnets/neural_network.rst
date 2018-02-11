
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
    │   ├── Apogee_CNN
    │   ├── StarNet2017
    │   └── Cifar10_CNN
    ├── BayesianCNNBase
    │   └── Apogee_BCNN
    ├── ConvVAEBase
    │   └── APGOEE_CVAE
    └── CGANBase
        └── GalaxyGAN2017

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
defined in `NeuralNetMaster` Class. `Normalizer` will not normalize data/labels equal to `magicnumber` defined in configuration file.
So that astroNN loss function can recognize those missing/bad data.

`Normalizer` consists of a few modes that you can, but the mode will minus mean and divide standard derivation to the data.

#. `Mode 0` means normalizing data with mean=0 and standard derivation=1 (same as doing nothing)
#. `Mode 1` means normalizing data with a single mean and a single standard derivation of the data
#. `Mode 2` means normalizing data with pixelwise means and pixelwise standard derivations of the data
#. `Mode 3` means normalizing data with a single mean and standard derivation=1 the data (only centered the data), it is useful for normalizing spectra
#. `Mode 255` means normalizing data with mean=127.5 and standard derivation=127.5, this mode is designed to normalize 8bit images

You can set the mode from a astroNN neural net instance before called `train()` method by

.. code-block:: python

    # To set the normalization mode for input and labels
    astronn_neuralnet.input_norm_mode = ...
    astronn_neuralnet.labels_norm_mode = ...

CNNBase Class
--------------------------------------
