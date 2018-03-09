# ---------------------------------------------------------#
#   astroNN.gaia.gaia_shared: shared functions for apogee
# ---------------------------------------------------------#

import os

import numpy as np
from astropy import units as u


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
    from astroNN import ENVVAR_WARN_FLAG
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
        print('dr is not provided, using default dr={}'.format(dr))
    else:
        pass
    return dr


def mag_to_fakemag(mag, parallax, parallax_err=None):
    """
    NAME:
        mag_to_fakemag
    PURPOSE:
        To convert appearant magnitude to astroNN's fake magnitude
    INPUT:
        mag (float, ndarray): appearant magnitude
        parallax (float, ndarray): parallax in mas
        parallax_err (float, ndarray): parallax err in mas
    OUTPUT:
        fakemag (float, ndarray)
        (conditional) fakemag_err (float, ndarray)
    HISTORY:
        2017-Oct-14 - Written - Henry Leung (University of Toronto)
    """

    # Check unit if available
    if type(parallax) == u.quantity.Quantity:
        original_parallax_unit = parallax.unit
        if parallax.unit != u.mas:
            parallax = parallax.to(u.mas)
            if parallax_err is not None:
                parallax_err = parallax_err * original_parallax_unit
                parallax_err = parallax_err.to(u.mas)
                parallax_err = parallax_err.value
            print(
                'Please be advised that astroNN fakemag function expects mas, astroNN has corrected the unit according'
                ' to astropy unit framework')
        # Take the value as we cant apply log10 to astropy unit
        parallax = parallax.value
    else:
        print('Please be advised that astroNN fakemag is parallax(mas) * 10 ** (0.2 * mag)')

    if parallax_err is None:
        return parallax * (10. ** (0.2 * mag))
    else:
        fakemag = parallax * (10. ** (0.2 * mag))
        fakemag_err = np.abs((parallax_err / parallax) * fakemag)
        return fakemag, fakemag_err


def mag_to_absmag(mag, parallax, parallax_err=None):
    """
    NAME:
        mag_to_absmag
    PURPOSE:
        To convert appearant magnitude to absolute magnitude
    INPUT:
        mag (float, ndarray): magnitude
        parallax (float, ndarray): parallax
        parallax_err (float, ndarray): parallax err in mas
    OUTPUT:
        absmag (float, ndarray)
        (conditional) absmag_err (float, ndarray)
    HISTORY:
        2017-Oct-14 - Written - Henry Leung (University of Toronto)
    """
    # Check unit if available
    if type(parallax) == u.quantity.Quantity:
        original_parallax_unit = parallax.unit
        if parallax.unit != u.arcsec:
            parallax = parallax.to(u.arcsec)
            if parallax_err is not None:
                parallax_err = parallax_err * original_parallax_unit
                parallax_err = parallax_err.to(u.arcsec)
                parallax_err = parallax_err.value
            print('Please be advised that astroNN mag_to_absmag() expects arcsecond, astroNN has corrected the unit '
                  'according to astropy unit framework')
        # Take the value as we cant apply log10 to astropy unit
        parallax = parallax.value
    else:
        print('Please be advised that astroNN mag_to_absmag expects parallax in (arcsecond)')

    if parallax_err is None:
        return mag + 5. * (np.log10(parallax) + 1.)
    else:
        absmag = mag + 5. * (np.log10(parallax) + 1.)
        absmag_err = 5. * np.abs(parallax_err / (parallax * np.log(10)))
        return absmag, absmag_err


def absmag_to_pc(absmag, mag):
    """
    NAME:
        absmag_to_pc
    PURPOSE:
        To convert absolute magnitude to parsec
    INPUT:
        mag (float, ndarray): magnitude
        absmag (float, ndarray): absolute magnitude
    OUTPUT:
        parsec (float, ndarray with astropy unit) in pc
    HISTORY:
        2017-Nov-16 - Written - Henry Leung (University of Toronto)
    """
    return (1. / (10. ** (((absmag - mag) / 5.) - 1.))) * u.parsec


def fakemag_to_absmag(fakemag):
    """
    NAME:
        fakemag_to_absmag
    PURPOSE:
        To convert fakemag to absmag
    INPUT:
        fakemag (float, ndarray): fakemag
    OUTPUT:
        absmag (float, ndarray
    HISTORY:
        2018-Jan-31 - Written - Henry Leung (University of Toronto)
    """
    return 5. * (np.log10(fakemag) - 2.)


def absmag_to_fakemag(absmag):
    """
    NAME:
        absmag_to_fakemag
    PURPOSE:
        To convert absmag to fakemag
    INPUT:
        fakemag (float, ndarray): fakemag
    OUTPUT:
        absmag (float, ndarray
    HISTORY:
        2018-Jan-31 - Written - Henry Leung (University of Toronto)
    """
    return 10. ** (0.2 * absmag + 2.)


def fakemag_to_pc(fakemag, mag, fakemag_err=None):
    """
    NAME:
        fakemag_to_absmag
    PURPOSE:
        To convert fakemag to parsec
    INPUT:
        fakemag (float, ndarray): fakemag
        mag (float, ndarray): magnitude
        fakemag_err (float, ndarray): fakemag err
    OUTPUT:
        parsec (float, ndarray with astropy unit) in pc
    HISTORY:
        2018-Jan-31 - Written - Henry Leung (University of Toronto)
    """
    if fakemag_err is None:
        return 1000. * (10. ** (0.2 * mag)) / fakemag * u.parsec
    else:
        pc = 1000. * (10. ** (0.2 * mag)) / fakemag * u.parsec
        pc_err = (fakemag_err / fakemag) * pc
        return pc, pc_err
