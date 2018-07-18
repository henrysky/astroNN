.. astroNN documentation master file, created by sphinx-quickstart on Thu Dec 21 17:52:45 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. warning:: This is a draft documentation aiming for latest astroNN commit

Welcome to astroNN's documentation!
======================================

astroNN is a python package to do various kinds of neural networks with targeted application in astronomy by using Keras
as model and training prototyping, but at the same time take advantage of Tensorflow's flexibility.

For non-astronomy applications, astroNN contains custom loss functions and layers which are compatible with Tensorflow
or Keras with Tensorflow backend. The custom loss functions mostly designed to deal with incomplete labels.
astroNN contains demo for implementing Bayesian Neural Net with Dropout Variational Inference in which you can get
reasonable uncertainty estimation and other neural nets.

For astronomy applications, astroNN contains some tools to deal with APOGEE, Gaia and LAMOST data. astroNN is mainly designed
to apply neural nets on APOGEE spectra analysis and predicting luminosity from spectra using data from Gaia
parallax with reasonable uncertainty from Bayesian Neural Net. Generally, astroNN can handle 2D and 2D colored images too.
Currently astroNN is a python package being developed by the main author to facilitate his research
project on deep learning application in stellar and galactic astronomy using SDSS APOGEE, Gaia and LAMOST data.

For learning purpose, astroNN includes a deep learning toy dataset for astronomer - :doc:`/galaxy10`.

Getting Started
---------------
astroNN is developed on GitHub. You can download astroNN from its Github_.

To clone the latest commit of astroNN from github

.. code-block:: bash

    $ git clone --depth=1 https://github.com/henrysky/astroNN

Recommended method of installation as astroNN is still in active development and will update daily:

.. code-block:: bash

    $ python setup.py develop

Or run the following command to install after you open a command line window in the package folder:

.. code-block:: bash

    $ python setup.py install

Or install via ``pip`` (Not recommended so far):
astroNN on `Python PyPI`_

.. code-block:: bash

    $ pip install astroNN

.. _Python PyPI: https://pypi.org/project/astroNN/

Indices, tables and astroNN structure
---------------------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

::

   astroNN/
   ├── apogee/
   │   ├── apogee_shared.py [shared codes across apogee module]
   │   ├── chips.py [functions to deal with apogee detectors and spectra]
   │   ├── downloader.py [functions to downlaod apogee data]
   │   └── plotting.py [functions to plot apogee data]
   ├── data/
   │   └──  ... [multiple pre-compiled data in numpy format]
   ├── datasets/
   │   ├──  apogee_distances.py
   │   ├──  apogee_rc.py
   │   ├──  apokasc.py
   │   ├──  galaxy10.py [astroNN's galaxy10 related codes]
   │   ├──  h5.py
   │   └──  xmatch.py [coordinates cross matching]
   ├── gaia/
   │   ├──  downloader.py [functions to downlaod gaia data]
   │   └──  gaia_shared.py [function related to astrometry and magnitude]
   ├── lamost/
   │   ├──  chips.py [functions to deal with lamost detectors and spectra]
   │   └──  lamost_shared.py [shared codes across lamost module]
   ├── models/ [contains neural network models]
   │   └──  ... [NN models codes and modules]
   ├── nn/
   │   ├──  callbacks.py [Keras's callbacks]
   │   ├──  layers.py [layers]
   │   ├──  losses.py [losses]
   │   ├──  metrics.py [metrics]
   │   └──  numpy.py [handy numpy tools]
   └── shared/ [shared codes across modules]

Datasets
--------------

* :doc:`/galaxy10`

Basics of astroNN
--------------------

.. toctree::
   :maxdepth: 2

   quick_start

   neuralnets/losses_metrics
   neuralnets/layers
   neuralnets/callback_utils
   neuralnets/basic_usage

Neural Net Introduction and Demonstration
-------------------------------------------

.. toctree::
   :maxdepth: 2

   neuralnets/BCNN
   gaia_dr2_special

* `Uncertainty Analysis of Neural Nets with Variational Methods`_
* `Galaxy10 Notebook`_
* :doc:`neuralnets/vae_demo`
* `Variational AutoEncoder with simple 1D data demo`_
* `Training neural net with DR14 APOGEE_Distances Value Added Catalogue using astroNN`_
* `Gaia DR2 things`_

APOGEE/Gaia/LAMOST Tools and Spectra Analysis using astroNN
-------------------------------------------------------------

.. toctree::
   :maxdepth: 2

   tools_apogee
   tools_lamost
   tools_gaia
   compile_datasets
   neuralnets/apogee_cnn
   neuralnets/apogee_bcnn
   neuralnets/apogee_bcnncensored
   neuralnets/ApogeeCVAE

Other Topics
----------------

* :doc:`/gaia_dr2_special`

* Astronomy Related Deep Learning Paper (Re)Implementation

  * :doc:`neuralnets/StarNet2017`

* :doc:`neuralnets/cifar10`
* :doc:`/history`
* `Known Issues`_

Acknowledging astroNN
-----------------------

Please cite astroNN in your publications if it helps your research. Here is an example BibTeX entry:

::

   @misc{leung2018astroNN,
     title={astroNN},
     author={Leung & Bovy},
     year={2018},
     howpublished={\url{https://github.com/henrysky/astroNN}},
   }

or AASTex

::

   \bibitem[Leung \& Bovy (2018)]{leung2018astroNN} Leung \& Bovy 2018, astroNN GitHub, https://github.com/henrysky/astroNN

Authors
-------------
-  | **Henry Leung** - *Initial work and developer* - henrysky_
   | Astronomy Student, University of Toronto
   | Contact Henry: henrysky.leung [at] mail.utoronto.ca

-  | **Jo Bovy** - *Project Supervisor* - jobovy_
   | Astronomy Professor, University of Toronto

.. _Github: https://github.com/henrysky/astroNN
.. _henrysky: https://github.com/henrysky
.. _jobovy: https://github.com/jobovy
.. _Uncertainty Analysis of Neural Nets with Variational Methods: https://github.com/henrysky/astroNN/tree/master/demo_tutorial/NN_uncertainty_analysis
.. _Variational AutoEncoder with simple 1D data demo: https://github.com/henrysky/astroNN/blob/master/demo_tutorial/VAE/variational_autoencoder_demo.ipynb
.. _Known Issues: https://github.com/henrysky/astroNN/issues
.. _Galaxy10 Notebook: https://github.com/henrysky/astroNN/blob/master/demo_tutorial/galaxy10/Galaxy10_Tutorial.ipynb
.. _Training neural net with DR14 APOGEE_Distances Value Added Catalogue using astroNN: https://github.com/henrysky/astroNN/blob/master/demo_tutorial/astroNN_in_action/apogee_distance_training.ipynb
.. _Gaia DR2 things: https://github.com/henrysky/astroNN/tree/master/demo_tutorial/gaia_dr1_dr2/