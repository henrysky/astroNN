.. image:: https://raw.githubusercontent.com/henrysky/astroNN/master/astroNN_icon_withname.png
   :width: 200px
   :align: center

|

.. image:: https://readthedocs.org/projects/astronn/badge/?version=latest
   :target: http://astronn.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/github/license/henrysky/astroNN.svg
   :target: https://github.com/henrysky/astroNN/blob/master/LICENSE
   :alt: GitHub license

.. image:: https://github.com/henrysky/astroNN/workflows/CI/badge.svg
   :target: https://github.com/henrysky/astroNN/actions
   :alt: Build Status

.. image:: https://codecov.io/gh/henrysky/astroNN/branch/master/graph/badge.svg?token=oI3JSmEHvG
  :target: https://codecov.io/gh/henrysky/astroNN

.. image:: https://badge.fury.io/py/astroNN.svg
    :target: https://badge.fury.io/py/astroNN

.. image:: http://img.shields.io/badge/DOI-10.1093/mnras/sty3217-blue.svg
   :target: http://dx.doi.org/10.1093/mnras/sty3217

Getting Started
=================

astroNN is a python package to do various kinds of neural networks with targeted application in astronomy by using Keras API
as model and training prototyping, but at the same time take advantage of Tensorflow's flexibility.

For non-astronomy applications, astroNN contains custom loss functions and layers which are compatible with Tensorflow. The custom loss functions mostly designed to deal with incomplete labels.
astroNN contains demo for implementing Bayesian Neural Net with Dropout Variational Inference in which you can get
reasonable uncertainty estimation and other neural nets.

For astronomy applications, astroNN contains some tools to deal with APOGEE, Gaia and LAMOST data. astroNN is mainly designed
to apply neural nets on APOGEE spectra analysis and predicting luminosity from spectra using data from Gaia
parallax with reasonable uncertainty from Bayesian Neural Net. Generally, astroNN can handle 2D and 2D colored images too.
Currently astroNN is a python package being developed by the main author to facilitate his research
project on deep learning application in stellar and galactic astronomy using SDSS APOGEE, Gaia and LAMOST data.

For learning purpose, astroNN includes a deep learning toy dataset for astronomer - `Galaxy10 Dataset`_.


`astroNN Documentation`_

`Quick Start guide`_

`Uncertainty Analysis of Neural Nets with Variational Methods`_


Acknowledging astroNN
-----------------------

| Please cite the following paper that describes astroNN if astroNN is used in your research as well as linking it to https://github.com/henrysky/astroNN
| **Deep learning of multi-element abundances from high-resolution spectroscopic data** [`arXiv:1808.04428`_][`ADS`_]

.. _arXiv:1808.04428: https://arxiv.org/abs/1808.04428
.. _ADS: https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.3255L/abstract

Authors
-------------
-  | **Henry Leung** - *Initial work and developer* - henrysky_
   | Astronomy Student, University of Toronto
   | Contact Henry: henrysky.leung [at] utoronto.ca

-  | **Jo Bovy** - *Project Supervisor* - jobovy_
   | Astronomy Professor, University of Toronto

License
-------------
This project is licensed under the MIT License - see the `LICENSE`_ file for details

.. _LICENSE: LICENSE
.. _henrysky: https://github.com/henrysky
.. _jobovy: https://github.com/jobovy

.. _astroNN Documentation: http://astronn.readthedocs.io/
.. _Quick Start guide: http://astronn.readthedocs.io/en/latest/quick_start.html
.. _Galaxy10 Dataset: http://astronn.readthedocs.io/en/latest/galaxy10.html
.. _Galaxy10 Tutorial Notebook: https://github.com/henrysky/astroNN/blob/master/demo_tutorial/galaxy10/Galaxy10_Tutorial.ipynb
.. _Uncertainty Analysis of Neural Nets with Variational Methods: https://github.com/henrysky/astroNN/tree/master/demo_tutorial/NN_uncertainty_analysis
