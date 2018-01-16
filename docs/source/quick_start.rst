
Getting Started
---------------
Recommended method of installation as this python package is still in active development and will update daily:

.. code-block:: bash

   $ python setup.py develop

Or run the following command to install after you open a command line window in the package folder:

.. code-block:: bash

   $ python setup.py install


Prerequisites
---------------
Anaconda 5.0.0 or above is recommended

::

    Python 3.6 or above
    Tensorflow OR Tensorflow-gpu (1.4.0 or above)
    Keras 2.1.2 or above
    CUDA 8.0 and CuDNN 6.1 (only neccessary for Tensorflow-gpu 1.4.0)
    CUDA 9.0 and CuDNN 7.0 (only neccessary for Tensorflow-gpu 1.5.0 RC)
    CUDA 9.1 is not supported!!
    graphviz and pydot_ng are required to plot the model architecture
    scikit-learn, tqdm and astroquery required for some basic astroNN function

For instruction on how to install Tensorflow, please refer to their
official website `Installing TensorFlow`_

`High Performance Tensorflow CPU MacOS build`_

Recommended system requirement:

::

    64-bits operating system
    CPU which supports AVX2 (Intel CPU 2014 or later, AMD CPU 2015 or later)
    8GB RAM or above
    Nvidia Graphics card (Optional, GTX900 series or above)
    (If using NVIDIA GPU): At least 2GB VRAM on GPU

.. _Installing TensorFlow: https://www.tensorflow.org/install/

.. _High Performance Tensorflow CPU MacOS build: https://github.com/lakshayg/tensorflow-build

.. note:: Only Keras with Tensorflow backend is tested and supported, issues with other backends will be ignored and won't fix.

.. note:: Multi-GPU or Intel/AMD graphics is not supported. Only Windows and Linux is officially supported by Tensorflow-GPU with compatible NVIDIA graphics


Folder Structure
-----------------

This code depends on an environment variables and folder. The
environment variables is ``SDSS_LOCAL_SAS_MIRROR``: top-level
directory that will be used to (selectively) mirror the SDSS SAS
``GAIA_TOOLS_DATA``: top-level directory under which the data will be
stored.

How to set environment variable on different operating system: `Guide
here`_

::

    NeuralNetMaster
    ├── CNNBase
    │   ├── Apogee_CNN
    │   ├── StarNet2017
    │   └── Cifar10
    ├── BayesianCNNBase
    │   └── Apogee_BCNN
    ├── ConvVAEBase
    │   └── APGOEE_CVAE
    └── CGANBase


    $SDSS_LOCAL_SAS_MIRROR/
    ├── dr14/
    │   ├── apogee/spectro/redux/r8/stars/
    │   │   ├── apo25m/
    │   │   │   ├── 4102/
    │   │   │   │   ├──  apStar-r8-2M21353892+4229507.fits
    │   │   │   │   ├──  apStar-r8-**********+*******.fits
    │   │   │   │   └──  ****/
    │   │   ├── apo1m/
    │   │   │   ├── hip/
    │   │   │   │   ├──  apStar-r8-2M00003088+5933348.fits
    │   │   │   │   ├──  apStar-r8-**********+*******.fits
    │   │   │   │   └──  ***/
    │   │   ├── l31c/l31c.2/
    │   │   │   ├── allStar-l30e.2.fits
    │   │   │   ├── allVisit-l30e.2.fits
    │   │   │   ├── 4102/
    │   │   │   │   ├──  aspcapStar-r8-l30e.2-2M21353892+4229507.fits
    │   │   │   │   ├──  aspcapStar-r8-l30e.2-**********+*******.fits
    │   │   │   │   └──  ****/
    │   │   │   ├── Cannon/
    │   │   │   │   └──  allStarCannon-l31c.2.fits
    ├── dr13/
    │   └── *similar to dr13/*


    $GAIA_TOOLS_DATA/
        gaia/tgas_source/fits/
            TgasSource_000-000-000.fits
            TgasSource_000-000-0**.fits

.. note:: The APOGEE and GAIA folder structure should be consistent with APOGEE_ and gaia_tools_ python package by Jo Bovy, tools for dealing with APOGEE and Gaia data

A dedicated project folder is recommended to run astroNN, always run astroNN under the root of project folder. So that astroNN will always create folder for every neural network you run under the same place. Just as below

.. image:: astronn_master_folder.PNG

.. _Guide here: https://www.schrodinger.com/kb/1842
.. _APOGEE: https://github.com/jobovy/apogee/
.. _gaia_tools: https://github.com/jobovy/gaia_tools/