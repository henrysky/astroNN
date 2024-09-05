
Getting Started
====================
``astroNN`` is developed on `Github`_ with its source code hosted there. But the easiest way to install is 
via ``pip`` on `Python PyPI`_ by running the following command in your terminal

.. prompt:: bash $

    pip install astroNN

.. _Github: https://github.com/henrysky/astroNN
.. _Python PyPI: https://pypi.org/project/astroNN/

To install the latest development version, you can clone the repository by running the following command:

.. prompt:: bash $

    git clone --depth=1 https://github.com/henrysky/astroNN.git

then install it by running:

.. prompt:: bash $

    python -m pip install .

or if you want to install it in editable mode, you can run:

.. prompt:: bash $

    python -m pip install -e .

Prerequisites
---------------

``astroNN`` requires Python 3.9 or above. The following packages are required which will be installed automatically when you install ``astroNN``:

.. literalinclude:: ../../requirements.txt

Currently ``astroNN`` supports both `Tensorflow`_ and `PyTorch`_ as backend. You can install either of them by running:

.. prompt:: bash $

    pip install tensorflow
    pip install torch

To plot the model, you will need to install ``graphviz``. On Ubuntu, you can install it by running:

.. prompt:: bash $

    sudo apt install graphviz

On MacOS, you can install it by running:

.. prompt:: bash $

    brew install graphviz

On Windows, you can download the Windows package from https://graphviz.gitlab.io/_pages/Download/Download_windows.html and add the package to the PATH environment variable.

.. _Tensorflow: https://www.tensorflow.org/install
.. _PyTorch: https://pytorch.org/get-started/locally/

Running on Google Colab
--------------------------------

To use the latest commit of astroNN on Google Colab, you can copy and paste the following 

.. prompt::

    !pip install torch keras
    !pip install git+https://github.com/henrysky/astroNN.git
    !export KERAS_BACKEND=torch

Basic FAQ
-----------------

How can I contribute?
+++++++++++++++++++++++++++++++

You can contact me (Henry: henrysky [dot] leung [at] utoronto [dot] ca) or refer to :doc:`/contributing`.

I have encountered a bug
+++++++++++++++++++++++++++++++++++++

Please try to use the latest commit of astroNN. If the issue persists, please report to https://github.com/henrysky/astroNN/issues

I am receiving warnings on missing environment variables
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

If you are not dealing with APOGEE or Gaia data, please ignore those warnings. If error raised to prevent you to use some
of astroNN functionality, please report it as a bug to https://github.com/henrysky/astroNN/issues

If you don't want those warnings to be shown again, go to astroNN's configuration file and set ``environmentvariablewarning``
to ``False``

I have installed `pydot` and `graphviz` but still fail to plot the model
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if you are encountering this issue, please uninstall both ``pydot`` and ``graphviz`` and run the following command

.. prompt:: bash $

    pip install pydot
    conda install graphviz

Then if you are using MacOS, run the following command

.. prompt:: bash $

   brew install graphviz

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

``magicnumber`` refers to the Magic Number which representing missing labels/data, default is ``nan``.

``multiprocessing_generator`` refers to whether enable multiprocessing in astroNN data generator. Default is False.

``environmentvariablewarning`` refers to whether you will be warned about not setting APOGEE and Gaia environment variable.

``custommodelpath`` refers to a list of custom models, path to the folder containing custom model (.py files),
multiple paths can be separated by ``;``.
Default value is `None` meaning no additional path will be searched when loading model.
Or for example: ``/users/astroNN/custom_models/;/local/some_other_custom_models/`` if you have self defined model in those locations.

``cpufallback`` refers to whether force keras to use CPU.

For whatever reason if you want to reset the configure file:

.. code-block:: python
    :linenos:

    from astroNN.config import config_path

    # astroNN will reset the config file if the flag = 2
    config_path(flag=2)


Dara Folder Structure and Environment Variables
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
   :scale: 50 %

.. _Guide here: https://www.schrodinger.com/kb/1842
.. _APOGEE: https://github.com/jobovy/apogee/
.. _gaia_tools: https://github.com/jobovy/gaia_tools/