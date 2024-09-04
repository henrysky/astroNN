.. automodule:: astroNN.gaia

Mini Tools for Gaia data
=============================================

.. note:: astroNN only contains a limited amount of necessary tools. For a more comprehensive python tool to deal with Gaia data, please refer to Jo Bovy's `gaia_tools`_

.. _gaia_tools: https://github.com/jobovy/gaia_tools

``astroNN.gaia`` module provides a handful of tools to deal with astrometry and photometry. 
The mission of the Gaia spacecraft is to create a dynamic, three-dimensional map of the Milky Way Galaxy by measuring
the distances, positions and proper motion of stars. To do this, the spacecraft employs two telescopes, an imaging
system, an instrument for measuring the brightness of stars, and a spectrograph. Launched in 2013, Gaia orbits the Sun
at Lagrange point L2, 1.5 million kilometres from Earth. By the end of its five-year mission, Gaia will have mapped well
over one billion stars—one percent of the Galactic stellar population.

*ESA Gaia satellite*: https://sci.esa.int/gaia/

.. automodule:: astroNN.gaia.downloader

fakemag (dummy scale)
-------------------------------

``fakemag`` is an astroNN dummy scale primarily used to preserve the gaussian standard error from Gaia. astroNN
always assume there is no error in apparent magnitude measurement.

:math:`L_\mathrm{fakemag} = \varpi 10^{\frac{1}{5}m_\mathrm{apparent}} = 10^{\frac{1}{5}M_\mathrm{absolute}+2}`, where
:math:`\varpi` is parallax in `mas`

You can get a sense of the fakemag scale from the following plot

.. image:: fakemag_scale.png
    :width: 500
    :align: center

Conversion Tools related to Astrometry and Magnitude
-----------------------------------------------------

Some functions have input error argument, they are optional and if you provided error, the function will propagate error
and have 2 returns (convened data, and converted propagated error), otherwise it will only has 1 return (converted data)

.. autofunction:: astroNN.gaia.mag_to_fakemag
.. autofunction:: astroNN.gaia.mag_to_absmag
.. autofunction:: astroNN.gaia.absmag_to_pc
.. autofunction:: astroNN.gaia.fakemag_to_absmag
.. autofunction:: astroNN.gaia.absmag_to_fakemag
.. autofunction:: astroNN.gaia.fakemag_to_pc
.. autofunction:: astroNN.gaia.fakemag_to_parallax
.. autofunction:: astroNN.gaia.fakemag_to_logsol
.. autofunction:: astroNN.gaia.absmag_to_logsol
.. autofunction:: astroNN.gaia.logsol_to_fakemag
.. autofunction:: astroNN.gaia.logsol_to_absmag
.. autofunction:: astroNN.gaia.fakemag_to_mag
.. autofunction:: astroNN.gaia.extinction_correction

All of these functions preserve ``magicnumber`` in input(s). 
Preserving ``magicnumber`` means the indices which matched ``magicnumber`` in ``config.ini`` will be preserved, for example:

.. code-block:: python

    >>> from astroNN.gaia import absmag_to_pc

    >>> absmag_to_pc([1., -9999.], [2., 1.])
    <Quantity [   15.84893192, -9999.        ] pc>

    >>> absmag_to_pc([1., -9999.], [-9999., 1.])
    <Quantity [-9999., -9999.] pc>

Since functions support astropy Quantity framework, you can convert between units easily. For example:

.. code-block:: python

    >>> from astroNN.gaia import absmag_to_pc
    >>> from astropy import units as u
    >>> import numpy as np

    >>> # Example data of [Vega, Sirius, Betelgeuse]
    >>> absmag = np.array([0.582, 1.42, -5.85])
    >>> mag = np.array([0.03, -1.46, 0.5])
    >>> pc = absmag_to_pc(absmag, mag)  # The output - pc - carries astropy unit

    >>> # Convert to AU
    >>> pc.to(u.AU)
    <Quantity [ 1599650.59975973,   547551.70190335, 38408304.24589768] AU>

    >>> # Or convert to angle units by using astropy's equivalencies function
    >>> pc.to(u.arcsec, equivalencies=u.parallax())
    <Quantity [0.12894366, 0.3767038 , 0.00537032] arcsec>

Since some functions support error propagation, lets say you are not familiar with ``fakemag`` and you want to know
how standard error in ``fakemag`` propagate to ``parsec``, you can for example

.. code-block:: python

    >>> from astroNN.gaia import fakemag_to_pc

    >>> fakemag = 300
    >>> fakemag_err = 100
    >>> apparent_mag = 10

    >>> fakemag_to_pc(fakemag, apparent_mag, fakemag_err)
    (<Quantity 333.33333333 pc>, <Quantity 111.11111111 pc>)


Coordinates Matching between catalogs xmatch
-------------------------------------------------------------

.. autofunction:: astroNN.datasets.xmatch.xmatch

Here is an example

.. code-block:: python

    >>> from astroNN.datasets import xmatch
    >>> import numpy as np

    >>> # Some coordinates for cat1, J2000.
    >>> cat1_ra = np.array([36.,68.,105.,23.,96.,96.])
    >>> cat1_dec = np.array([72.,56.,54.,55.,88.,88.])

    >>> # Some coordinates for cat2, J2000.
    >>> cat2_ra = np.array([23.,56.,222.,96.,245.,68.])
    >>> cat2_dec = np.array([36.,68.,82.,88.,26.,56.])

    >>> # Using maxdist=2 arcsecond separation threshold, because its default, so not shown here
    >>> # Using epoch1=2000. and epoch2=2000., because its default, so not shown here
    >>> # because both datasets are J2000., so no need to provide pmra and pmdec which represent proper motion
    >>> idx_1, idx_2, sep = xmatch(ra1=cat1_ra, dec1=cat1_dec, ra2=cat2_ra, dec2=cat2_dec)
    >>> idx_1
    array([1, 4, 5])
    >>> idx_2
    array([5, 3, 3])
    >>> cat1_ra[idx_1], cat2_ra[idx_2]
    (array([68., 96., 96.]), array([68., 96., 96.]))

    >>> # What happens if we swap cat_1 and cat_2
    >>> idx_1, idx_2, sep = xmatch(ra1=cat2_ra, dec1=cat2_dec, ra2=cat1_ra, dec2=cat1_dec)
    >>> idx_1
    array([3, 5])
    >>> idx_2
    array([4, 1])
    >>> cat1_ra[idx_2], cat2_ra[idx_1]
    (array([96., 68.]), array([96., 68.]))
