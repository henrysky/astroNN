
Getting Started
====================
astroNN is developed on GitHub. You can download astroNN from its Github_.

But the easiest way to install is via ``pip``: astroNN on `Python PyPI`_

.. code-block:: bash

    $ pip install astroNN

.. _Github: https://github.com/henrysky/astroNN
.. _Python PyPI: https://pypi.org/project/astroNN/

For latest version, you can clone the latest commit of astroNN from github

.. code-block:: bash

    $ git clone --depth=1 https://github.com/henrysky/astroNN

and run the following command to install after you open a command line window in the package folder:

.. code-block:: bash

    $ python setup.py install

Prerequisites
---------------
Latest version of Anaconda is recommended, but generally the use of Anaconda is still highly recommended

::

    Python 3.6 or above
    Tensorflow OR Tensorflow-gpu (the latest version is recommended)
    Tensorflow-Probability (the latest version is recommended)
    CUDA and CuDNN (only necessary for Tensorflow-gpu)
    graphviz and pydot are required to plot the model architecture
    scikit-learn, tqdm and astroquery required for some basic astroNN function

Since `Tensorflow`_ and `Tensorflow-Probability`_ are rapidly developing packages and astroNN heavily depends on Tensorflow.
The support policy of astroNN to these packages is only the last 2 official versions are supported (i.e. the latest
and the previous version are included in test suite). Generally the latest version of Tensorflow, Tensorflow-Probability and
optional Keras is recommended. The current supporting status (i.e. included in test cases) are

.. _`Tensorflow`: https://github.com/tensorflow/tensorflow
.. _`Tensorflow-Probability`: https://github.com/tensorflow/probability

::

    Tensorflow OR Tensorflow-gpu 1.14.x (correspond to Tensorflow-Probability 0.7.0)
    Tensorflow OR Tensorflow-gpu 1.13.x (correspond to Tensorflow-Probability 0.6.0)
    Tensorflow OR Tensorflow-gpu 1.12.x (correspond to Tensorflow-Probability 0.5.0)

.. note::

    Due to bugs in Tensorflow/Tensorflow-gpu, for 1.12.x: https://github.com/tensorflow/tensorflow/issues/22952 and for
    1.14.x: https://github.com/tensorflow/tensorflow/issues/27543, you have to patch a few lines in order for astroNN to work proporly.

    You can patch Tensorflow by running the following code

    .. code-block:: python

       from astroNN.config import tf_patch

       tf_patch()

    You can also unpatch Tensorflow to undo changes made by astroNN by running the following code

    .. code-block:: python

       from astroNN.config import tf_unpatch

       tf_unpatch()

For instruction on how to install Tensorflow, please refers to their official website `Installing TensorFlow`_

If you install `tensorflow` instead of `tensorflow-gpu`, Tensorflow will run on CPU. Currently official Tensorflow
python wheels do not compiled with AVX2 - sets of CPU instruction extensions that can speed up calculation on CPU.
If you are using `tensorflow` which runs on CPU only or want to use latest CUDA/CuDNN , you should consider to download
`High Performance Tensorflow MacOS build`_ for MacOS, Or `High Performance Tensorflow Windows build`_ for Windows.

Recommended system requirement:

::

    64-bits operating system
    CPU which supports AVX2 (List of CPUs: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX2)
    8GB RAM or above
    NVIDIA Graphics card (Optional, GTX900 series or above)
    (If using NVIDIA GPU): At least 2GB VRAM on GPU

.. _Installing TensorFlow: https://www.tensorflow.org/install/

.. _High Performance Tensorflow MacOS build: https://github.com/lakshayg/tensorflow-build

.. _High Performance Tensorflow Windows build: https://github.com/fo40225/tensorflow-windows-wheel

.. note:: Multi-GPU, Intel/AMD graphics are not supported. Only Windows and Linux are officially supported by Tensorflow-GPU with compatible NVIDIA graphics

Basic FAQ
-----------------

My hardware or software cannot meet the prerequisites, what should I do?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The hardware and software requirement is just an estimation. It is entirely possible to run astroNN without those
requirement. But generally, python 3.6 or above (as Tensorflow only supports py36 or above) and mid-to-high end hardware.

Can I contribute to astroNN?
+++++++++++++++++++++++++++++++

You can contact me (Henry: henrysky.leung [at] mail.utoronto.ca) or refer to :doc:`/contributing`.

I have found a bug in astorNN
+++++++++++++++++++++++++++++++++

Please try to use the latest commit of astroNN. If the issue persists, please report to https://github.com/henrysky/astroNN/issues

I keep receiving warnings on APOGEE and Gaia environment variables
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

If you are not dealing with APOGEE or Gaia data, please ignore those warnings. If error raised to prevent you to use some
of astroNN functionality, please report it as a bug to https://github.com/henrysky/astroNN/issues

If you don't want those warnings to be shown again, go to astroNN's configuration file and set ``environmentvariablewarning``
to ``False``

I have installed `pydot` and `graphviz` but still fail to plot the model
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if you are encountering this issue, please uninstall both ``pydot`` and ``graphviz`` and run the following command

.. code-block:: bash

    $ pip install pydot
    $ conda install graphviz

Then if you are using Mac, run the following command

.. code-block:: bash

   $ brew install graphviz

If you are using Windows, go to https://graphviz.gitlab.io/_pages/Download/Download_windows.html to download the Windows
package and add the package to the PATH environment variable.

.. automodule:: astroNN.config

Configuration File
---------------------

astroNN configuration file is located at ``~/.astroNN/config.ini`` which contains a few astroNN settings.

Currently, the default configuration file should look like this

::

    [Basics]
    magicnumber = -9999.0
    multiprocessing_generator = False
    environmentvariablewarning = True

    [NeuralNet]
    custommodelpath = None
    cpufallback = False
    gpu_mem_ratio = True

``magicnumber`` refers to the Magic Number which representing missing labels/data, default is -9999. Please do not change
this value if you rely on APOGEE data.

``multiprocessing_generator`` refers to whether enable multiprocessing in astroNN data generator. Default is False
except on Linux and MacOS.

``environmentvariablewarning`` refers to whether you will be warned about not setting APOGEE and Gaia environment variable.

``custommodelpath`` refers to a list of custom models, path to the folder containing custom model (.py files),
multiple paths can be separated by ``;``.
Default value is `None` means no path. Or for example: ``/users/astroNN/custom_models/;/local/some_other_custom_models/``

``cpufallback`` refers to whether force to use CPU. No effect if you are using tensorflow instead of tensorflow-gpu

``gpu_mem_ratio`` refers to GPU management. Set ``True`` to dynamically allocate memory which is astroNN default or enter a float between 0 and 1
to set the maximum ratio of GPU memory to use or set ``None`` to let Tensorflow pre-occupy all of available GPU memory
which is a designed default behavior from Tensorflow.

For whatever reason if you want to reset the configure file:

.. code-block:: python

   from astroNN.config import config_path

   # astroNN will reset the config file if the flag = 2
   config_path(flag=2)


Folder Structure for astroNN, APOGEE, Gaia and LAMOST data
----------------------------------------------------------

This code depends on environment variables and folders for APOGEE, Gaia and LAMOST data. The environment variables are

- ``SDSS_LOCAL_SAS_MIRROR``: top-level directory that will be used to (selectively) mirror the SDSS Science Archive Server (SAS)
- ``GAIA_TOOLS_DATA``: top-level directory under which the Gaia data will be stored.
- ``LASMOT_DR5_DATA``: top-level directory under which the LASMOST DR5 data will be stored.

How to set environment variable on different operating system: `Guide here`_

::

    $SDSS_LOCAL_SAS_MIRROR/
    ├── dr14/
    │   ├── apogee/spectro/redux/r8/stars/
    │   │   ├── apo25m/
    │   │   │   ├── 4102/
    │   │   │   │   ├── apStar-r8-2M21353892+4229507.fits
    │   │   │   │   ├── apStar-r8-**********+*******.fits
    │   │   │   │   └── ****/
    │   │   ├── apo1m/
    │   │   │   ├── hip/
    │   │   │   │   ├── apStar-r8-2M00003088+5933348.fits
    │   │   │   │   ├── apStar-r8-**********+*******.fits
    │   │   │   │   └── ***/
    │   │   ├── l31c/l31c.2/
    │   │   │   ├── allStar-l30e.2.fits
    │   │   │   ├── allVisit-l30e.2.fits
    │   │   │   ├── 4102/
    │   │   │   │   ├── aspcapStar-r8-l30e.2-2M21353892+4229507.fits
    │   │   │   │   ├── aspcapStar-r8-l30e.2-**********+*******.fits
    │   │   │   │   └── ****/
    │   │   │   └── Cannon/
    │   │   │       └── allStarCannon-l31c.2.fits
    └── dr13/
        └── *similar to dr14 above/*


    $GAIA_TOOLS_DATA/
    └── Gaia/
        ├── gdr1/tgas_source/fits/
        │   ├── TgasSource_000-000-000.fits
        │   ├── TgasSource_000-000-001.fits
        │   └── ***.fits
        └── gdr2/gaia_source_with_rv/fits/
            ├── GaiaSource_2851858288640_1584379458008952960.fits
            ├── GaiaSource_1584380076484244352_2200921635402776448.fits
            └── ***.fits

    $LASMOT_DR5_DATA/
    └── DR5/
        ├── LAMO5_2MS_AP9_SD14_UC4_PS1_AW_Carlin_M.fits
        ├── 20111024
        │   ├── F5902
        │   │   ├──spec-55859-F5902_sp01-001.fits.gz
        │   │   └── ****.fits.gz
        │   └── ***/
        ├── 20111025
        │   ├── B6001
        │   │   ├──spec-55860-B6001_sp01-001.fits.gz
        │   │   └── ****.fits.gz
        │   └── ***/
        └── ***/

.. note:: The APOGEE and Gaia folder structure should be consistent with APOGEE_ and gaia_tools_ python package by Jo Bovy, tools for dealing with APOGEE and Gaia data

A dedicated project folder is recommended to run astroNN, always run astroNN under the root of project folder. So that
astroNN will always create folder for every neural network you run under the same place. Just as below

.. image:: astronn_master_folder.PNG

.. _Guide here: https://www.schrodinger.com/kb/1842
.. _APOGEE: https://github.com/jobovy/apogee/
.. _gaia_tools: https://github.com/jobovy/gaia_tools/