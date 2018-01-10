.. astroNN documentation master file, created by
   sphinx-quickstart on Thu Dec 21 17:52:45 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Mini Tools for GAIA data
===========================

.. note:: astroNN only contains a limited amount of neccessary tools. For a more comprehansive python tool to deal with Gaia data, please refer to Jo Bovy's `gaia_tools`_


.. _gaia_tools: https://github.com/jobovy/gaia_tools


TGAS donwnload and loader
----------------------------

.. code:: python

    from astroNN.gaia import tgas
    from astroNN.gaia import tgas_load

    # To download tgas dr1 to GAIA_TOOLS_DATA and it will return the list of path to those files
    files_paths = tgas(dr=1)

    # To load the tgas dr1 files and return ra(J2015), dec(J2015), pmra, pmdec, parallax, parallax error, g-band mag
    ra, dec, pmra, pmdec, par, par_var,g_mag = tgas_load(dr=1)


Anderson et al 2017 improved parallax from data-driven stars model
-------------------------------------------------------------------------

.. code:: python

    from astroNN.gaia import anderson_2017_parallax

    # To load the improved parallax
    ra, dec, parallax, para_var = anderson_2017_parallax()