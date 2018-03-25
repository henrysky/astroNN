
Getting Started
====================
To clone the latest commit of astroNN from github

.. code-block:: bash

   $ git clone --depth=1 git://github.com/henrysky/astroNN

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

Prerequisites
---------------
Anaconda 5.0.0 or above is recommended, but generally the use of Anaconda is highly recommended

::

    Python 3.6 or above
    Tensorflow OR Tensorflow-gpu (1.5.0 or above)
    Keras 2.1.3 or above (Optional but recommended, Must be configured Tensorflow as backends)
    CUDA 9.0 and CuDNN 7.0 (only neccessary for Tensorflow-gpu 1.5.0)
    CUDA 9.1 is not supported!!
    graphviz and pydot_ng are required to plot the model architecture
    scikit-learn, tqdm and astroquery required for some basic astroNN function

For instruction on how to install Tensorflow, please refers to their
official website `Installing TensorFlow`_

Although Keras is optional, but its highly recommended. For instruction on how to install Keras, please refers to their
official website `Installing Keras`_

If you install `tensorflow` instead of `tensorflow-gpu`, Tensorflow will run on CPU. Currently official Tensorflow
python wheels do not compiled with AVX2 - sets of CPU instruction extensions that can speed up calculation on CPU.
If you are using `tensorflow` which runs on CPU only , you should consider to download
`High Performance Tensorflow MacOS build`_ for MacOS, Or `High Performance Tensorflow Windows build`_ for Windows.

Recommended system requirement:

::

    64-bits operating system
    CPU which supports AVX2 (Intel CPU 2014 or later, AMD CPU 2015 or later)
    8GB RAM or above
    NVIDIA Graphics card (Optional, GTX900 series or above)
    (If using NVIDIA GPU): At least 2GB VRAM on GPU

.. _Installing TensorFlow: https://www.tensorflow.org/install/

.. _Installing Keras: https://keras.io/#installation

.. _High Performance Tensorflow MacOS build: https://github.com/lakshayg/tensorflow-build

.. _High Performance Tensorflow Windows build: https://github.com/fo40225/tensorflow-windows-wheel

.. note:: Multi-GPU or Intel/AMD graphics is not supported. Only Windows and Linux is officially supported by Tensorflow-GPU with compatible NVIDIA graphics

Basic FAQ
-----------------

My hardware or software cannot meet the prerequisites, what should I do?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The hardware and software requirement is just an estimation. It is entirely possible to run astroNN without those
requirement. But generally, python 3.6 or above (as Tensorflow only supports py36 or above) and reasonably modern hardware.

Can I contribute to astroNN?
+++++++++++++++++++++++++++++++

Yes, you can contact me (Henry: henrysky.leung [at] mail.utoronto.ca) and tell me your idea

I have found a bug in astorNN
+++++++++++++++++++++++++++++++++

Please try to use the latest commit of astroNN. If the issue persists, please report to https://github.com/henrysky/astroNN/issues

I keep receiving warnings on APOGEE and Gaia environment variables
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

If you are not dealing with APOGEE or Gaia data, please ignore those warnings. If error raised to prevent you to use some
of astroNN functionality, please report it as a bug to https://github.com/henrysky/astroNN/issues

If you don't want those warnings to be shown again, go to astroNN's configuration file and set ``environmentvariablewarning``
to ``False``

I have installed `pydot_ng` and `graphviz` but still fail to plot the model
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if you are encountering this issue, please uninstall both ``pydot_ng`` and ``graphviz`` and run the following command

.. code-block:: bash

    $ pip install pydot_ng
    $ conda install graphviz

Configuration File
---------------------

astroNN configuration file is located at ``~/.astroNN/config.ini`` which contains a few astroNN settings.

Currently, the configuration file should look like this

::

    [Basics]
    magicnumber = -9999.0
    multiprocessing_generator = False
    environmentvariablewarning = True
    tensorflow_keras = auto

    [NeuralNet]
    custommodelpath = None
    cpufallback = False
    gpu_mem_ratio = True

``magicnumber`` refers to the Magic Number which representing missing labels/data, default is -9999.

``multiprocessing_generator`` refers to whether enable multiprocessing in astroNN data generator. Default is False
except on Linux and MacOS.

``environmentvariablewarning`` refers to whether you will be warned about not setting APOGEE and Gaia environment variable.

``tensorflow_keras`` refers to whether use `keras` or `tensorflow.keras`. Default option is ``auto`` to let astroNN
to decide (`keras` always be considered first), ``tensorflow`` to force it to use `tensorflow.keras` or ``keras`` to
force it to use `keras`

``custommodelpath`` refers to a list of custom models, path to the folder containing custom model (.py files),
multiple paths can be separated by ``;``.
Default value is `None` means no path. Or for example: ``/users/astroNN/custom_models/;/local/some_other_custom_models/``

``cpufallback`` refers to whether force to use CPU. No effect if you are using tensorflow instead of tensorflow-gpu

``gpu_mem_ratio`` refers to GPU management. Set ``True`` to dynamically allocate memory or enter a float smaller then 1
to set the maximum ratio of GPU memory to use or set ``None`` to let astroNN pre-occupy all of available GPU memory

For whatever reason if you want to reset the configure file:

.. code-block:: python

   from astroNN.config import config_path

   # astroNN will reset the config file if the flag = 2
   config_path(flag=2)


Folder Structure for astroNN, APOGEE and Gaia data
---------------------------------------------------

This code depends on an environment variables and folder for APOGEE and Gaia data. The
environment variables is ``SDSS_LOCAL_SAS_MIRROR``: top-level
directory that will be used to (selectively) mirror the SDSS SAS
``GAIA_TOOLS_DATA``: top-level directory under which the data will be
stored.

How to set environment variable on different operating system: `Guide
here`_

::

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
    │   │   │   └── Cannon/
    │   │   │       └──  allStarCannon-l31c.2.fits
    └── dr13/
        └── *similar to dr14 above/*


    $GAIA_TOOLS_DATA/
    └── gaia/tgas_source/fits/
        ├── TgasSource_000-000-000.fits
        ├── TgasSource_000-000-001.fits
        └── ***/

.. note:: The APOGEE and Gaia folder structure should be consistent with APOGEE_ and gaia_tools_ python package by Jo Bovy, tools for dealing with APOGEE and Gaia data

A dedicated project folder is recommended to run astroNN, always run astroNN under the root of project folder. So that astroNN will always create folder for every neural network you run under the same place. Just as below

.. image:: astronn_master_folder.PNG

.. _Guide here: https://www.schrodinger.com/kb/1842
.. _APOGEE: https://github.com/jobovy/apogee/
.. _gaia_tools: https://github.com/jobovy/gaia_tools/