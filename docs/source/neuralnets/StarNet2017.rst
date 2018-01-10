.. astroNN documentation master file, created by
   sphinx-quickstart on Thu Dec 21 17:52:45 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

StarNet (arXiv:1709.09182)
---------------------------

StarNet2017 is a astroNN neural network implementation from the paper (`arXiv:1709.09182`_), StarNet2017 is inherited from
astroNN's CNNBase class defined in astroNN.models.NeuralNetBases

You can create StarNet2017 via

.. code:: python

    from astroNN.models import StarNet2017

    # And then create an object of StarNet2017 classs
    starnet_net = StarNet2017()

.. note:: Default hyperparameter is the same as the original StarNet paper

.. _arXiv:1709.09182: https://arxiv.org/abs/1709.09182
