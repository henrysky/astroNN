.. astroNN documentation master file, created by
sphinx-quickstart on Thu Dec 21 17:52:45 2017.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive.

Mini Tools for APOGEE data
=============================

.. note:: astroNN only contains a limited amount of neccessary tools. For a more comprehansive python tool to deal with APOGEE data, please refer to Jo Bovy's `APOGEE tools`_


.. _APOGEE tools: hhttps://github.com/jobovy/apogee


Pesudo-Continuum Normalization of Spectra
==========================================

You can access the default astroNN continuum mask by

.. code:: python

   import os
   import astroNN
   import numpy as np

   dr = 14

   dir = os.path.join(os.path.dirname(astroNN.__path__[0]), 'astroNN', 'data', 'dr{}_contmask.npy'.format(dr))
   cont_mask = np.load(dir)


When you do normalization using astroNN, you can just use con_mask=None to use default mask

.. code:: python

   from astroNN.apogee import continuum

   spec, spec_var = continuum(spectra, spectra_vars, cont_mask=None, deg=2, dr=14)

.. note:: If you are planning to compile APOGEE dataset using astroNN, you can ignore this section as astroNN H5Compiler will load data from fits files directly and will take care everything.

.. image:: con_mask_spectra.png

APOGEE data downloaders
=======================

astroNN apogee data downloader always act as functions that will return you the path of downloaded file(s), and download it if it does not exist locally.

-----------------------------------
General way to open the fits file
-----------------------------------

astropy.io.fits documentation: http://docs.astropy.org/en/stable/io/fits/

.. code:: python

   from astropy.io import fits

   data = fits.open(local_path_to_file)

--------------
 allstar file
--------------

Data Model: https://data.sdss.org/datamodel/files/APOGEE_REDUX/APRED_VERS/APSTAR_VERS/ASPCAP_VERS/RESULTS_VERS/allStar.html

.. code:: python

   from astroNN.apogee import allstar

   local_path_to_file = allstar(dr=14)


------------------------------
Combined spectra (aspcapStar)
------------------------------

Data Model: https://data.sdss.org/datamodel/files/APOGEE_REDUX/APRED_VERS/APSTAR_VERS/ASPCAP_VERS/RESULTS_VERS/LOCATION_ID/aspcapStar.html

.. code:: python

   from astroNN.apogee import combined_spectra

   local_path_to_file = combined_spectra(dr=14, location=a_location_id, apogee=a_apogee_id)

------------------------------
Visit spectra (apStar)
------------------------------

Data Model: https://data.sdss.org/datamodel/files/APOGEE_REDUX/APRED_VERS/APSTAR_VERS/TELESCOPE/LOCATION_ID/apStar.html

.. code:: python

   from astroNN.apogee import visit_spectra

   local_path_to_file = visit_spectra(dr=14, location=a_location_id, apogee=a_apogee_id)

-----------------------------------------
Red Clumps of SDSS Value Added Catalogs
-----------------------------------------

Data Model (DR14): https://data.sdss.org/datamodel/files/APOGEE_RC/cat/apogee-rc-DR14.html

.. code:: python

   from astroNN.apogee.downloader import apogee_vac_rc

   local_path_to_file = apogee_vac_rc(dr=14)

-----------------------------------------
APOKASC in the Kepler Fields
-----------------------------------------

.. code:: python

   from astroNN.datasets.apokasc import apokasc_load

   gold_ra, gold_dec, gold_logg, basic_ra, basic_dec, basic_logg = apokasc_load()