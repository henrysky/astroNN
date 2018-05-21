.. automodule:: astroNN.models.StarNet2017

StarNet (arXiv:1709.09182)
---------------------------

StarNet2017 is a astroNN neural network implementation from the paper (`arXiv:1709.09182`_), StarNet2017 is inherited from
astroNN's CNNBase class defined in astroNN.models.NeuralNetBases

You can create StarNet2017 via

.. code:: python

    from astroNN.models import StarNet2017
    from astroNN.datasets import H5Loader

    # And then create an object of StarNet2017 classs
    starnet_net = StarNet2017()

    # Load the train data from dataset first, x_train is spectra and y_train will be ASPCAP labels
    loader = H5Loader('datasets.h5')
    loader.load_err = False
    x_train, y_train = loader.load()

    # And then create an object of Convolutional Neural Network classs
    starnet = StarNet2017()

    # Set max_epochs to 10 for a quick result. You should train more epochs normally
    starnet.max_epochs = 10
    starnet.train(x_train, y_train)

.. note:: Default hyperparameter is the same as the original StarNet paper

.. _arXiv:1709.09182: https://arxiv.org/abs/1709.09182
