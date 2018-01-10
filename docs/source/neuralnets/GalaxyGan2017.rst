.. astroNN documentation master file, created by
   sphinx-quickstart on Thu Dec 21 17:52:45 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. warning:: Currently not working!

GalaxyGAN (arXiv:1702.00403)
---------------------------------

GalaxyGAN2017 is a astroNN neural network implementation from the paper (`arXiv:1702.00403`_), GalaxyGAN2017 is inherited
from astroNN's CGANBase class defined in astroNN.models.NeuralNetBases

You can create GalaxyGAN2017 via

.. code:: python

    from astroNN.models import GalaxyGAN2017

    # And then create an object of GalaxyGAN2017 classs
    galaxygan_net = GalaxyGAN2017()

.. note:: Default hyperparameter is the same as the original GalaxyGAN paper

.. _arXiv:1702.00403: https://arxiv.org/abs/1702.00403
