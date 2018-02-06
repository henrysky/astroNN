.. astroNN documentation master file, created by sphinx-quickstart on Thu Dec 21 17:52:45 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. warning:: This is a draft documentation aiming for latest astroNN commit

Welcome to astroNN's documentation!
======================================

astroNN is a python package to do various kind of neural networks for astronomers.

Besides conventional neural network like convolutional neural net, astroNN provides bayesian neural network
implementation to do neural network with incomplete labeled data and uncertainty analysis.
Incomplete labeled data means you have some target labels, but you only has a subset of them for some data. astroNN
will look for MAGIC_NUMBER (Default is -9999.) in training data and wont backpropagate those particular labels for
those particular data. For uncertainty analysis, please see the demonstration section.

Furthermore, astroNN also included a deep learning toy dataset for astronomer - Galaxy10.

As of now, this is a python package developing for an undergraduate research project on deep learning application in
stellar and galactic astronomy using SDSS APOGEE DR14 and Gaia DR1.

Getting Started
---------------
astroNN is developed on GitHub. You can download astroNN from its Github_.

Recommended method of installation as this python package is still in active development and will update daily:

.. code-block:: bash

   $ python setup.py develop

Or run the following command to install after you open a command line window in the package folder:

.. code-block:: bash

   $ python setup.py install

Datasets
--------------
* :doc:`/galaxy10`

List of Tutorial and Documentation
----------------------------------------

.. toctree::
   :maxdepth: 2

   quick_start

   tools_apogee
   tools_gaia
   compile_datasets
   neuralnets/neural_network
   neuralnets/losses_metrics
   neuralnets/custom_layers
   neuralnets/CNN
   neuralnets/BCNN
   neuralnets/Conv_VAE
   neuralnets/apogee_cnn
   neuralnets/apogee_bcnn


* Astronomy Related Deep Learning Paper (Re)Implementation

  * :doc:`neuralnets/StarNet2017`
  * :doc:`neuralnets/GalaxyGan2017`

* :doc:`neuralnets/cifar_galaxy10`
* :doc:`/history`
* `Known Issues`_

Demonstration
-----------------
* :doc:`neuralnets/vae_demo`
* `Uncertainty Analysis in Bayesian Deep Learning Demonstration`_
* `Variational AutoEncoder with simple 1D data demo`_

Authors
-------------
-  | **Henry Leung** - *Initial work and developer* - henrysky_
   | Astronomy Undergrad, University of Toronto
   | Contact Henry: henrysky.leung [at] mail.utoronto.ca

-  | **Jo Bovy** - *Supervisor of Henry Leung* - jobovy_
   | Astronomy Professor, University of Toronto

-  | :doc:`/acknowledgments`

Indices and tables
----------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Github: https://github.com/henrysky/astroNN
.. _henrysky: https://github.com/henrysky
.. _jobovy: https://github.com/jobovy
.. _Uncertainty Analysis in Bayesian Deep Learning Demonstration: https://github.com/henrysky/astroNN/tree/master/demo_tutorial/NN_uncertainty_analysis
.. _Variational AutoEncoder with simple 1D data demo: https://github.com/henrysky/astroNN/blob/master/demo_tutorial/VAE/variational_autoencoder_demo.ipynb
.. _Known Issues: https://github.com/henrysky/astroNN/issues