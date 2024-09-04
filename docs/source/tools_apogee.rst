
.. automodule:: astroNN.apogee

Mini Tools for APOGEE data
=================================================

.. note:: astroNN only contains a limited amount of necessary tools. For a more comprehensive python tool to deal with APOGEE data, please refer to Jo Bovy's `APOGEE tools`_

.. _APOGEE tools: hhttps://github.com/jobovy/apogee

``astroNN.apogee`` module has a handful of tool to deal with APOGEE data.
The APO Galactic Evolution Experiment employs high-resolution, high signal-to-noise infrared spectroscopy
to penetrate the dust that obscures significant fractions of the disk and bulge of our Galaxy. APOGEE is surveying 
red giant stars across the full range of the Galactic bulge, bar, disk, and halo. APOGEE generated precise
radial velocities and detailed chemical abundances, providing unprecedented insights into the dynamical structure and
chemical history of the Galaxy. In conjunction with the planet-finding surveys, Kepler and CoRoT, APOGEE unravels
problems in fundamental astrophysics.

*SDSS APOGEE*: https://www.sdss.org/surveys/apogee-2/

.. automodule:: astroNN.apogee.chips

Continuum Normalization of APOGEE Spectra
------------------------------------------------

You can access the default astroNN continuum mask fro APOGEE spectra by

.. code-block:: python
   :linenos:

   import os
   import astroNN
   import numpy as np

   dr = 14

   dir = os.path.join(os.path.dirname(astroNN.__path__[0]), 'astroNN', 'data', f'dr{dr}_contmask.npy')
   cont_mask = np.load(dir)


When you do continuum normalization using astroNN, you can just use con_mask=None to use default mask provided by Jo Bovy's APOGEE Tools.
astroNN will use a SINGLE continuum pixel mask to normalize all spectra you provided. Moreover, astroNN will normalize
the spectra by chips instead of normalize them all together.

.. autofunction::  astroNN.apogee.apogee_continuum

.. code-block:: python
   :linenos:

   from astroNN.apogee import apogee_continuum

   # spectra_errs refers to the 1-sigma error array provided by APOGEE
   # spectra can be multiple spectra at a time
   norm_spec, norm_spec_err = apogee_continuum(apogee_spectra, spectra_errs, cont_mask=None, deg=2, dr=14)

   # If you deal with bitmask too and want to set some target bits to zero, you can add additional arguement in apogee_continuum()
   # You target_bit=[a list of number] or target_bit=None to use default target_bit
   apogee_continuum(apogee_spectra, spectra_errs, cont_mask=None, deg=2, dr=14, bitmask=apogee_bitmask, target_bit=None)

`norm_spec` refers to the normalized spectra while `norm_spec_err` refers to the normalized spectra error

.. image:: con_mask_spectra.png
   :scale: 50 %

You can use ``continuum()`` to normalize any spectra while ``apogee_continuum()`` is specifically designed for APOGEE spectra.

.. code-block:: python
   :linenos:

   from astroNN.apogee import continuum

   spec, spec_err = continuum(spectra, spectra_errs, cont_mask, deg=2)


Basics Tools related to APOGEE Spectra
--------------------------------------------

Here are some basic tools to deal with APOGEE spectra

-------------------------------------------------
Retrieve Basic APOGEE Spectra Pixel Information
-------------------------------------------------

You can retrieve basic APOGEE spectra pixel information by

.. autofunction::  astroNN.apogee.chips_pix_info

.. code-block:: python
   :linenos:

   from astroNN.apogee import chips_pix_info

   info = chips_pix_info(dr=14)

   # info[0] refers to the location where blue chips starts
   # info[1] refers to the location where blue chips ends
   # info[2] refers to the location where green chips starts
   # info[3] refers to the location where blue chips end
   # info[4] refers to the location where red chips starts
   # info[5] refers to the location where red chips ends
   # info[6] refers to the total number of pixels after deleting gap

------------------------------------
APOGEE Spectra Wavelength Solution
------------------------------------

.. autofunction::  astroNN.apogee.wavelength_solution

You can retrieve APOGEE spectra wavelength solution by

.. code-block:: python
   :linenos:

   from astroNN.apogee import wavelength_solution

   lambda_blue, lambda_green, lambda_red = wavelength_solution(dr=16)

   # lambda_blue refers to the wavelength solution for each pixel in blue chips
   # lambda_green refers to the wavelength solution for each pixel in green chips
   # lambda_red refers to the wavelength solution for each pixel in red chips

------------------------------------
APOGEE Spectra Gap Delete
------------------------------------

.. autofunction::  astroNN.apogee.gap_delete

You can delete the gap between raw spectra by

.. code-block:: python
   :linenos:

   from astroNN.apogee import gap_delete

   # original_spectra can be multiple spectra at a time
   gap_deleted_spectra = gap_delete(original_spectra, dr=16)

------------------------------------------
Split APOGEE Spectra into Three Detectors
------------------------------------------

.. autofunction::  astroNN.apogee.chips_split

You can split APOGEE spectra into three detectors by

.. code-block:: python
   :linenos:

   from astroNN.apogee import chips_split

   # original_spectra can be multiple spectra at a time
   spectra_blue, spectra_green, spectra_red = chips_split(original_spectra, dr=16)

`chips_split()` will delete the gap between the detectors if you give raw APOGEE spectra. If you give gap deleted spectra,
then the function will simply split the spectra into three.

------------------------------------
APOGEE Bitmask to Boolean Array
------------------------------------

You can turn a APOGEE PIXMASK bitmask array into a boolean array provided you have some target bit you want to mask

Bitmask: https://www.sdss.org/dr16/algorithms/bitmasks/#collapseAPOGEE_PIXMASK

.. autofunction::  astroNN.apogee.bitmask_boolean

Example:

.. code-block:: python
   :linenos:

   from astroNN.apogee import bitmask_boolean
   import numpy as np

   spectra_bitmask = np.array([2048, 128, 1024, 512, 16, 8192, 4096, 64, 2, 32, 256, 8, 4, 16896])
   boolean_output = bitmask_boolean(spectra_bitmask, target_bit=[0,1,2,3,4,5,6,7,9,12])
   print(boolean_output)
   # array([[False, True, False, True, True, False, True, True, True, True, False, True, True, True]])

-----------------------------------------------
Decompose APOGEE Bitmask into Constitute Bits
-----------------------------------------------

You can turn a APOGEE PIXMASK bit into its constitute bits

Bitmask: https://www.sdss.org/dr16/algorithms/bitmasks/#collapseAPOGEE_PIXMASK

.. autofunction::  astroNN.apogee.bitmask_decompositor

.. code-block:: python
   :linenos:

   from astroNN.apogee import bitmask_decompositor

   decomposed_bits = bitmask_decompositor(single_bitmask)

Example:

.. code-block:: python
   :linenos:

   from astroNN.apogee import bitmask_decompositor

   # Create a simulated bit number
   # Lets say this pixel is marked as 0, 5, 13 and 14 bit
   bitmask = 2**0 + 2**5 + 2**13 + 2**14

   decomposed_bits = bitmask_decompositor(bitmask)
   # The function returns the set of original bits
   # array([ 0,  5, 13, 14])

-----------------------------------------------
Retrieve ASPCAP Elements Window Mask
-----------------------------------------------

Original ASPCAP Elements Windows Mask: https://svn.sdss.org/public/repo/apogee/idlwrap/trunk/lib/ which is described in
https://arxiv.org/abs/1502.04080

You can get ASPCAP elements window mask as a boolean array by providing an element name to this function,

.. autofunction:: astroNN.apogee.aspcap_mask

.. code-block:: python
   :linenos:

   from astroNN.apogee import aspcap_mask

   mask = aspcap_mask('Mg')  # for example you want to get ASPCAP Mg mask

APOGEE Data Downloader
---------------------------

.. automodule:: astroNN.apogee.downloader

astroNN APOGEE data downloader always act as functions that will return you the path of downloaded file(s),
and download it if it does not exist locally. If the file cannot be found on server, astroNN will generally return ``False`` as the path.

--------------------------------
General Way to Open Fits File
--------------------------------

astropy.io.fits documentation: https://docs.astropy.org/en/stable/io/fits/

.. code-block:: python
   :linenos:

   from astropy.io import fits

   data = fits.open(local_path_to_file)

--------------
allstar file
--------------

Data Model: https://data.sdss.org/datamodel/files/APOGEE_ASPCAP/APRED_VERS/ASPCAP_VERS/allStar.html

.. autofunction::  astroNN.apogee.allstar

.. code-block:: python
   :linenos:

   from astroNN.apogee import allstar

   local_path_to_file = allstar(dr=16)

---------------
allvisit file
---------------

Data Model: https://data.sdss.org/datamodel/files/APOGEE_ASPCAP/APRED_VERS/ASPCAP_VERS/allVisit.html

.. autofunction::  astroNN.apogee.allvisit

.. code-block:: python
   :linenos:

   from astroNN.apogee import allvisit

   local_path_to_file = allvisit(dr=16)

------------------------------
Combined Spectra (aspcapStar)
------------------------------

Data Model: https://data.sdss.org/datamodel/files/APOGEE_ASPCAP/APRED_VERS/ASPCAP_VERS/TELESCOPE/FIELD/aspcapStar.html

.. autofunction:: astroNN.apogee.combined_spectra

.. code-block:: python
   :linenos:

   from astroNN.apogee import combined_spectra

   local_path_to_file = combined_spectra(dr=16, location=a_location_id, apogee=a_apogee_id)

------------------------------
Visit Spectra (apStar)
------------------------------

Data Model: https://data.sdss.org/datamodel/files/APOGEE_REDUX/APRED_VERS/stars/TELESCOPE/FIELD/apStar.html

.. autofunction:: astroNN.apogee.visit_spectra

.. code-block:: python
   :linenos:

   from astroNN.apogee import visit_spectra

   local_path_to_file = visit_spectra(dr=16, location=a_location_id, apogee=a_apogee_id)

-----------------------------------------
astroNN catalogue for APOGEE
-----------------------------------------

Introduction: https://www.sdss.org/dr16/data_access/value-added-catalogs/?vac_id=the-astronn-catalog-of-abundances,-distances,-and-ages-for-apogee-dr16-stars

Data Model (DR16): https://data.sdss.org/datamodel/files/APOGEE_ASTRONN/apogee_astronn.html

.. autofunction:: astroNN.apogee.downloader.apogee_astronn

.. code-block:: python
   :linenos:

   from astroNN.apogee import apogee_astronn

   local_path_to_file = apogee_astronn(dr=16)


-----------------------------------------
Red Clumps of SDSS Value Added Catalogs
-----------------------------------------

Introduction: https://www.sdss.org/dr16/data_access/value-added-catalogs/?vac_id=apogee-red-clump-rc-catalog

Data Model (DR16): https://data.sdss.org/datamodel/files/APOGEE_RC/cat/apogee-rc-DR16.html

.. autofunction:: astroNN.datasets.apogee.load_apogee_rc

.. code-block:: python
   :linenos:

   from astroNN.apogee import apogee_rc

   local_path_to_file = apogee_rc(dr=16)

Or you can use `load_apogee_rc()` to load the data by

.. code-block:: python
   :linenos:

   from astroNN.datasets import load_apogee_rc

   # unit can be 'distance' for distance in parsec, 'absmag' for k-band absolute magnitude
   # 'fakemag' for astroNN's k-band fakemag scale
   RA, DEC, array = load_apogee_rc(dr=16, unit='distance', extinction=True)  # extinction only effective if not unit='distance'

-----------------------------------------
APOGEE Distance Estimations
-----------------------------------------

Introduction: https://www.sdss.org/dr14/data_access/value-added-catalogs/?vac_id=apogee-dr14-based-distance-estimations

Data Model (DR14): https://data.sdss.org/datamodel/files/APOGEE_DISTANCES/apogee_distances.html
Data Model (DR16): https://data.sdss.org/datamodel/files/APOGEE_STARHORSE/apogee_starhorse.html

.. autofunction:: astroNN.apogee.apogee_distances

.. code-block:: python
   :linenos:

   from astroNN.apogee.downloader import apogee_distances

   local_path_to_file = apogee_distances(dr=14)

.. autofunction:: astroNN.datasets.load_apogee_distances

Or you can use `load_apogee_distances()` to load the data by

.. code-block:: python
   :linenos:

   from astroNN.datasets import load_apogee_distances

   # unit can be 'distance' for distance in parsec, 'absmag' for k-band absolute magnitude
   # 'fakemag' for astroNN's k-band fakemag scale
   # cuts=True to cut out those unknown values (-9999.) and measurement error > 20%
   RA, DEC, array, err_array = load_apogee_distances(dr=14, unit='distance', cuts=True, keepdims=False)

--------------------
Cannon's allstar
--------------------

Introduction: https://data.sdss.org/datamodel/files/APOGEE_REDUX/APRED_VERS/APSTAR_VERS/ASPCAP_VERS/RESULTS_VERS/CANNON_VERS/cannonModel.html

Data Model (DR14): https://data.sdss.org/datamodel/files/APOGEE_REDUX/APRED_VERS/APSTAR_VERS/ASPCAP_VERS/RESULTS_VERS/CANNON_VERS/allStarCannon.html

.. autofunction::  astroNN.apogee.allstar_cannon

.. code-block:: python
   :linenos:

   from astroNN.apogee import allstar_cannon

   local_path_to_file = allstar_cannon(dr=14)
