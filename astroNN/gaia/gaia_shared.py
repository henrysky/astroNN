# ---------------------------------------------------------#
#   astroNN.gaia.gaia_shared: shared functions for apogee
# ---------------------------------------------------------#

import os
import warnings

import numpy as np
from astropy import units as u
from astroNN.config import MAGIC_NUMBER

default_parallax_unit = u.mas


def gaia_env():
    """
    NAME:
        gaia_env
    PURPOSE:
        get Gaia enviroment variable
    INPUT:
    OUTPUT:
        (path)
    HISTORY:
        2017-Oct-26 - Written - Henry Leung (University of Toronto)
    """
    from astroNN.config import ENVVAR_WARN_FLAG
    _GAIA = os.getenv('GAIA_TOOLS_DATA')
    if _GAIA is None and ENVVAR_WARN_FLAG is True:
        print("WARNING! Gaia environment variable GAIA_TOOLS_DATA not set")
    return _GAIA


def gaia_default_dr(dr=None):
    """
    NAME:
        gaia_default_dr
    PURPOSE:
        Check if dr arguement is provided, if none then use default
    INPUT:
        dr (int): GAIA DR, example dr=1
    OUTPUT:
        dr (int): GAIA DR, example dr=1
    HISTORY:
        2017-Oct-26 - Written - Henry Leung (University of Toronto)
    """
    if dr is None:
        dr = 1
        print(f'dr is not provided, using default dr={dr}')
    else:
        pass
    return dr


def mag_to_fakemag(mag, parallax, parallax_err=None):
    """
    NAME:
        mag_to_fakemag
    PURPOSE:
        To convert appearant magnitude to astroNN's fake magnitude
        Magic Number will be preserved
    INPUT:
        mag (float, ndarray): appearant magnitude
        parallax (float, ndarray, astropy.units.quantity): parallax in mas
        parallax_err (float, ndarray, astropy.units.quantity): parallax err in mas
    OUTPUT:
        fakemag (float, ndarray)
        (conditional) fakemag_err (float, ndarray)
    HISTORY:
        2017-Oct-14 - Written - Henry Leung (University of Toronto)
    """
    # Check unit if available
    if type(parallax) == u.quantity.Quantity:
        original_parallax_unit = parallax.unit
    else:
        print(f'Please be advised that astroNN fakemag is parallax({default_parallax_unit.name}) * 10 ** (0.2 * mag)')
        original_parallax_unit = default_parallax_unit

    mag = np.array(mag)
    parallax = np.array(parallax)

    magic_idx = ((parallax == MAGIC_NUMBER) | (mag == MAGIC_NUMBER))  # check for magic number
    parallax = parallax * original_parallax_unit

    if parallax.unit != default_parallax_unit:
        parallax = parallax.to(default_parallax_unit)
        if parallax_err is not None:
            parallax_err = parallax_err * original_parallax_unit if type(parallax_err) != u.quantity.Quantity else \
                parallax_err
            parallax_err = parallax_err.to(default_parallax_unit)
            parallax_err = parallax_err.value
        print(
            f'Please be advised that astroNN fakemag function expects {default_parallax_unit.name}, astroNN has '
            f'corrected the unit according to astropy unit framework')
    # Take the value as we cant apply pow() to astropy unit
    parallax = parallax.value

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBEr
        warnings.simplefilter("ignore")
        fakemag = parallax * (10. ** (0.2 * mag))
    if type(parallax) == np.ndarray:  # check if its only 1 element
        fakemag[magic_idx] = MAGIC_NUMBER
    else:
        fakemag = MAGIC_NUMBER if magic_idx == [1] else fakemag

    if parallax_err is None:
        return fakemag
    else:
        fakemag_err = np.abs((parallax_err / parallax) * fakemag)
        if type(parallax) == np.ndarray:  # check if its only 1 element
            fakemag_err[magic_idx] = MAGIC_NUMBER
        else:
            fakemag_err = MAGIC_NUMBER if magic_idx == [1] else fakemag_err
        return fakemag, fakemag_err


def mag_to_absmag(mag, parallax, parallax_err=None):
    """
    NAME:
        mag_to_absmag
    PURPOSE:
        To convert appearant magnitude to absolute magnitude
        Magic Number will be preserved
    INPUT:
        mag (float, ndarray): magnitude
        parallax (float, ndarray, astropy.units.quantity): parallax in mas
        parallax_err (float, ndarray, astropy.units.quantity): parallax err in mas
    OUTPUT:
        absmag (float, ndarray)
        (conditional) absmag_err (float, ndarray)
    HISTORY:
        2017-Oct-14 - Written - Henry Leung (University of Toronto)
    """

    # Check unit if available
    if type(parallax) == u.quantity.Quantity:
        original_parallax_unit = parallax.unit
    else:
        print(f'Please be advised that astroNN mag_to_absmag expects parallax in {default_parallax_unit.name}')
        original_parallax_unit = default_parallax_unit

    mag = np.array(mag)
    parallax = np.array(parallax)

    magic_idx = ((parallax == MAGIC_NUMBER) | (mag == MAGIC_NUMBER))  # check for magic number
    parallax = parallax * original_parallax_unit

    if parallax.unit != default_parallax_unit:
        parallax = parallax.to(default_parallax_unit)
        if parallax_err is not None:
            parallax_err = parallax_err * original_parallax_unit if type(parallax_err) != u.quantity.Quantity else \
                parallax_err
            parallax_err = parallax_err.to(default_parallax_unit)
            parallax_err = parallax_err.value
        print(f'Please be advised that astroNN mag_to_absmag() expects {default_parallax_unit.name}, '
              f'astroNN has corrected the unit according to astropy unit framework')
    # Take the value as we cant apply log10 to astropy unit
    parallax = parallax.value

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBEr
        warnings.simplefilter("ignore")
        absmag = mag + 5. * (np.log10(parallax) - 2.)

    if type(parallax) == np.ndarray:  # check if its only 1 element
        absmag[magic_idx] = MAGIC_NUMBER
    else:
        absmag = MAGIC_NUMBER if magic_idx == [1] else absmag
    if parallax_err is None:
        return absmag
    else:
        absmag_err = 5. * np.abs(parallax_err / (parallax * absmag))
        if type(parallax) == np.ndarray:  # check if its only 1 element
            absmag_err[magic_idx] = MAGIC_NUMBER
        else:
            absmag_err = MAGIC_NUMBER if magic_idx == [1] else absmag_err
        return absmag, absmag_err


def absmag_to_pc(absmag, mag):
    """
    NAME:
        absmag_to_pc
    PURPOSE:
        To convert absolute magnitude to parsec
        Magic Number will be preserved
    INPUT:
        mag (float, ndarray): magnitude
        absmag (float, ndarray): absolute magnitude
    OUTPUT:
        parsec (float, ndarray with astropy unit) in pc
    HISTORY:
        2017-Nov-16 - Written - Henry Leung (University of Toronto)
    """
    absmag = np.array(absmag)
    mag = np.array(mag)
    magic_idx = ((absmag == MAGIC_NUMBER) | (mag == MAGIC_NUMBER))  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBEr
        warnings.simplefilter("ignore")
        pc = (1. / (10. ** (((absmag - mag) / 5.) - 1.))) * u.parsec

    if type(absmag) == np.ndarray:  # check if its only 1 element
        pc[magic_idx] = MAGIC_NUMBER * u.parsec
        return pc
    else:
        return MAGIC_NUMBER if magic_idx == [1] else pc


def fakemag_to_absmag(fakemag):
    """
    NAME:
        fakemag_to_absmag
    PURPOSE:
        To convert fakemag to absmag
        Magic Number will be preserved
    INPUT:
        fakemag (float, ndarray): fakemag
    OUTPUT:
        absmag (float, ndarray
    HISTORY:
        2018-Jan-31 - Written - Henry Leung (University of Toronto)
    """
    fakemag = np.array(fakemag)
    magic_idx = (fakemag == MAGIC_NUMBER)  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBEr
        warnings.simplefilter("ignore")
        absmag = 5. * (np.log10(fakemag) - 2.)

    if type(absmag) == np.ndarray:  # check if its only 1 element
        absmag[magic_idx] = MAGIC_NUMBER
    else:  # for float
        absmag = MAGIC_NUMBER if magic_idx == [1] else absmag
    return absmag


def absmag_to_fakemag(absmag):
    """
    NAME:
        absmag_to_fakemag
    PURPOSE:
        To convert absmag to fakemag
        Magic Number will be preserved
    INPUT:
        fakemag (float, ndarray): fakemag
    OUTPUT:
        absmag (float, ndarray
    HISTORY:
        2018-Jan-31 - Written - Henry Leung (University of Toronto)
    """
    absmag = np.array(absmag)
    magic_idx = (absmag == MAGIC_NUMBER)  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBEr
        warnings.simplefilter("ignore")
        fakemag = 10. ** (0.2 * absmag + 2.)
    if type(fakemag) == np.ndarray:  # check if its only 1 element
        fakemag[magic_idx] = MAGIC_NUMBER
    else:  # for float
        fakemag = MAGIC_NUMBER if magic_idx == [1] else fakemag
    return fakemag


def fakemag_to_pc(fakemag, mag, fakemag_err=None):
    """
    NAME:
        fakemag_to_absmag
    PURPOSE:
        To convert fakemag to parsec
        Magic Number will be preserved
    INPUT:
        fakemag (float, ndarray): fakemag
        mag (float, ndarray): magnitude
        fakemag_err (float, ndarray): fakemag err
    OUTPUT:
        parsec (float, ndarray with astropy unit) in pc
    HISTORY:
        2018-Jan-31 - Written - Henry Leung (University of Toronto)
    """
    fakemag = np.array(fakemag)
    mag = np.array(mag)
    magic_idx = ((fakemag == MAGIC_NUMBER) | (mag == MAGIC_NUMBER))  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBEr
        warnings.simplefilter("ignore")
        pc = 1000. * (10. ** (0.2 * mag)) / fakemag
    if type(fakemag) == np.ndarray:  # check if its only 1 element
        pc[magic_idx] = MAGIC_NUMBER
    else:  # for float
        pc = MAGIC_NUMBER if magic_idx == [1] else pc

    if fakemag_err is None:
        return pc * u.parsec
    else:
        pc_err = (fakemag_err / fakemag) * pc
        if type(fakemag) == np.ndarray:  # check if its only 1 element
            pc_err[magic_idx] = MAGIC_NUMBER
        else:  # for float
            pc_err = MAGIC_NUMBER if magic_idx == [1] else pc_err
        return pc * u.parsec, pc_err * u.parsec
