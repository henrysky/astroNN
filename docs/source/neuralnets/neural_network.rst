
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
folder for every run

Normalizer
---------------

astroNN `Normalizer` is called when `train()` method is called and involved `pre_training_checklist_master()` method
defined in `NeuralNetMaster` Class

`Normalizer` consists of a few mode

#. `Normalizer.normalization_mode=0` means normalizing data with mean=0 and standard derivation=1 (doing nothing)
#. `Normalizer.normalization_mode=1`
#. `Normalizer.normalization_mode=2`
#. `Normalizer.normalization_mode=3`
#. `Normalizer.normalization_mode=255` means normalizing data with mean=127.5 and standard derivation=127.5, this mode is designed to normalize 8bit images



CNNBase Class
--------------------------------------
