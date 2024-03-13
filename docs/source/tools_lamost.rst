
.. automodule:: astroNN.lamost

Mini Tools for LAMOST data
=================================================

``astroNN.lamost`` module is designed for dealing with LAMOST DR5.

**LAMOST DR5 is not a public data release yet, this module only provides a limited amount of tools to deal with the spectra.
If you do not have the data, astroNN will not provide any LAMOST DR5 data nor functions to download them.**

*LAMOST Data Policy*: https://www.lamost.org/policies/data_policy.html

*LAMOST DR5 Homepage*: https://dr5.lamost.org/

*LAMOST DR5 Data Model*: https://dr5.lamost.org/doc/data-production-description

LAMOST Spectra Wavelength Solution
------------------------------------

.. autofunction::  astroNN.lamost.wavelength_solution

You can retrieve LAMOST spectra wavelength solution by

.. code:: python

    from astroNN.lamost import wavelength_solution

    lambda_solution = wavelength_solution(dr=5)

Pseudo-Continuum Normalization of LAMOST Spectra
---------------------------------------------------

.. autofunction::  astroNN.lamost.pseudo_continuum

.. code:: python

    from astroNN.lamost import pseudo_continuum

    # spectra_errs refers to the inverse variance array provided by LAMOST
    # spectra can be multiple spectra at a time
    norm_spec, norm_spec_err = pseudo_continuum(spectra, spectra_errs, dr=5)

Load LAMOST DR5 catalogue
---------------------------

.. autofunction::  astroNN.lamost.load_allstar_dr5

.. code:: python

    from astroNN.lamost import load_allstar_dr5

    fits_file = load_allstar_dr5()
    fits_file[1].header  # print file header
