.. astroNN documentation master file, created by sphinx-quickstart on Thu Dec 21 17:52:45 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to astroNN's documentation!
======================================

astroNN is a python package to do various kinds of neural networks with targeted application in astronomy by using Keras API
as model and training prototyping, but at the same time take advantage of Tensorflow or PyTorch flexibility.

For non-astronomy applications, astroNN contains custom loss functions and layers which are compatible with Keras v3. The custom loss functions mostly designed to deal with incomplete labels.
astroNN contains demo for implementing Bayesian Neural Net with Dropout Variational Inference in which you can get
reasonable uncertainty estimation and other neural nets.

For astronomy applications, astroNN contains some tools to deal with APOGEE, Gaia and LAMOST data. astroNN is mainly designed
to apply neural nets on APOGEE spectra analysis and predicting luminosity from spectra using data from Gaia
parallax with reasonable uncertainty from Bayesian Neural Net. Generally, astroNN can handle 2D and 2D colored images too.
Currently astroNN is a python package being developed by the main author to facilitate his research
project on deep learning application in stellar and galactic astronomy using SDSS APOGEE, Gaia and LAMOST data.

For learning purpose, astroNN includes a deep learning toy dataset for astronomer - :doc:`/galaxy10`.

Acknowledging astroNN
-----------------------

| Please cite the following paper that describes astroNN if astroNN is used in your research as well as linking it to https://github.com/henrysky/astroNN
| **Deep learning of multi-element abundances from high-resolution spectroscopic data** [`arXiv:1808.04428`_][`ADS`_]

.. _arXiv:1808.04428: https://arxiv.org/abs/1808.04428
.. _ADS: https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.3255L/abstract

Here is a list of publications using ``astroNN`` - :doc:`papers`

Authors
-------------
-  | **Henry Leung** - *Initial work and developer* - henrysky_
   | Department of Astronomy & Astrophysics, University of Toronto
   | Contact Henry: henrysky.leung [at] utoronto.ca

-  | **Jo Bovy** - *Project Supervisor* - jobovy_
   | Department of Astronomy & Astrophysics, University of Toronto

.. _Github: https://github.com/henrysky/astroNN
.. _henrysky: https://github.com/henrysky
.. _jobovy: https://github.com/jobovy
.. _Uncertainty Analysis of Neural Nets with Variational Methods: https://github.com/henrysky/astroNN/tree/master/demo_tutorial/NN_uncertainty_analysis
.. _Variational AutoEncoder with simple 1D data demo: https://github.com/henrysky/astroNN/blob/master/demo_tutorial/VAE/variational_autoencoder_demo.ipynb
.. _Galaxy10 Notebook: https://github.com/henrysky/astroNN/blob/master/demo_tutorial/galaxy10/Galaxy10_Tutorial.ipynb
.. _Training neural net with DR14 APOGEE_Distances Value Added Catalogue using astroNN: https://github.com/henrysky/astroNN/blob/master/demo_tutorial/astroNN_in_action/apogee_distance_training.ipynb
.. _Gaia DR2 things: https://github.com/henrysky/astroNN/tree/master/demo_tutorial/gaia_dr1_dr2/

Indices, tables and astroNN structure
---------------------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
   :maxdepth: 1
   :caption: Datasets

   galaxy10
   galaxy10sdss

.. toctree::
   :maxdepth: 2
   :caption: Basics of astroNN

   quick_start
   contributing
   history
   papers
   neuralnets/losses_metrics
   neuralnets/layers
   neuralnets/callback_utils
   neuralnets/neuralODE
   neuralnets/basic_usage

.. toctree::
   :maxdepth: 2
   :caption: NN Introduction and Demo

   neuralnets/BCNN

* `Uncertainty Analysis of Neural Nets with Variational Methods`_
* `Galaxy10 Notebook`_
* `Variational AutoEncoder with simple 1D data demo`_

.. toctree::
   :maxdepth: 2
   :caption: APOGEE/Gaia/LAMOST Tools and models

   tools_apogee
   tools_lamost
   tools_gaia
   compile_datasets
   neuralnets/apogee_cnn
   neuralnets/apogee_bcnn
   neuralnets/apogee_bcnncensored
   neuralnets/apogeedr14_gaiadr2_bcnn
   neuralnets/apogee_cvae
   neuralnets/apokasc_encoder
   neuralnets/StarNet2017
   neuralnets/cifar10
