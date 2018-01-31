
Mini Tools for GAIA data
===========================

.. note:: astroNN only contains a limited amount of neccessary tools. For a more comprehansive python tool to deal with Gaia data, please refer to Jo Bovy's `gaia_tools`_


.. _gaia_tools: https://github.com/jobovy/gaia_tools


General way to open fits file
--------------------------------

astropy.io.fits documentation: http://docs.astropy.org/en/stable/io/fits/

.. code:: python

   from astropy.io import fits

   data = fits.open(local_path_to_file)

TGAS downloader and loader
----------------------------

.. code:: python

    from astroNN.gaia import tgas
    from astroNN.gaia import tgas_load

    # To download tgas dr1 to GAIA_TOOLS_DATA and it will return the list of path to those files
    files_paths = tgas(dr=1)

    # To load the tgas DR1 files and return ra(J2015), dec(J2015), pmra, pmdec, parallax, parallax error, g-band mag
    ra, dec, pmra, pmdec, par, par_var,g_mag = tgas_load(dr=1)

Gaia_source downloader
-----------------------------------

.. code:: python

    from astroNN.gaia import gaia_source

    # To download gaia_source DR1 to GAIA_TOOLS_DATA and it will return the list of path to those files
    files_paths = gaia_source(dr=1)

Anderson et al 2017 improved parallax from data-driven stars model
-------------------------------------------------------------------------

.. code:: python

    from astroNN.gaia import anderson_2017_parallax

    # To load the improved parallax
    ra, dec, parallax, para_var = anderson_2017_parallax()

fakemag (astroNN dummy scale)
-------------------------------

``fakemag`` is an astroNN dummy scale primarily used to preserve the gaussian error from GAIA satellite.

:math:`M_{fakemag} = \omega 10^{\frac{1}{5}M_{apparent}} = 10^{\frac{1}{5}M_{absolute}+2}`, where
:math:`\omega` is parallax in `mas`



Conversion Tools related to astrometry
---------------------------------------

``mag_to_fakemag(mag, parallax)`` takes parallax in mas and apparent magnitude to astroNN's fakemag

``mag_to_absmag(mag, parallax)`` takes parallax in arcsec and apparent magnitude to astroNN's fakemag

``absmag_to_pc(absmag, mag)`` takes absolute magnitude and apparent magnitude to parsec, output is an numpy array with corresponding astropy Quantity

``fakemag_to_absmag(fakemag)``  takes fakemag to absolute magnitude

``fakemag_to_pc(fakemag, mag)``  takes fakemag and apparent magnitude to parsec, output is an numpy array with corresponding astropy Quantity

All of these functions can be imported by

.. code:: python

    from astroNN.gaia import ...

Since some functions support astropy Quantity framework, you can convert between units easily. Example:

.. code:: python

    from astroNN.gaia import absmag_to_pc
    from astropy import units as u
    import numpy as np

    # Example data of [Vega, Sirius, Betelgeuse]
    absmag = np.array([0.582, 1.42, -5.85])
    mag = np.array([0.03, -1.46, 0.5])
    pc = absmag_to_pc(absmag, mag)  # The output - pc - carries astropy unit

    # Convert to AU
    distance_in_AU = pc.to(u.AU)

    # Or convert to angle units by using astropy's equivalencies function
    arcsec = pc.to(u.arcsec, equivalencies=u.parallax())
