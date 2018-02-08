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
        get GAIA enviroment variable
    INPUT:
    OUTPUT:
        (path)
    HISTORY:
        2017-Oct-26 - Written - Henry Leung (University of Toronto)
    """
    _GAIA = os.getenv('GAIA_TOOLS_DATA')
    if _GAIA is None:
        print("WARNING! Gaia enviroment variable GAIA_TOOLS_DATA not set")
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


def mag_to_fakemag(mag, parallax):
    """
    NAME:
        mag_to_fakemag
    PURPOSE:
        To convert appearant magnitude to astroNN's fake magnitude
    INPUT:
        mag (float, ndarray): appearant magnitude
        parallax (float, ndarray): parallax in mas
    OUTPUT:
        fakemag (float)
    HISTORY:
        2017-Oct-14 - Written - Henry Leung (University of Toronto)
    """

    # Check unit if available
    if type(parallax) == u.quantity.Quantity:
        if parallax.unit != u.mas:
            parallax.to(u.mas)
            print(
                'Please be advised that astroNN fakemag function expects mas, astroNN has corrected the unit according'
                ' to astropy unit framework')
        # Take the value as we cant apply log10 to astropy unit
        parallax = parallax.value
    else:
        print('Please be advised that astroNN fakemag is parallax(mas) * 10 ** (0.2 * mag)')
    return parallax * (10 ** (0.2 * mag))


def mag_to_absmag(mag, parallax):
    """
    NAME:
        mag_to_absmag
    PURPOSE:
        To convert appearant magnitude to absolute magnitude
    INPUT:
        mag (float, ndarray): magnitude
        parallax (float, ndarray): parallax
    OUTPUT:
        absmag (float)
    HISTORY:
        2017-Oct-14 - Written - Henry Leung (University of Toronto)
    """
    # Check unit if available
    if type(parallax) == u.quantity.Quantity:
        if parallax.unit != u.arcsec:
            parallax.to(u.arcsec)
            print('Please be advised that astroNN mag_to_absmag() expects arcsecond, astroNN has corrected the unit '
                  'according to astropy unit framework')
        # Take the value as we cant apply log10 to astropy unit
        parallax = parallax.value
    else:
        print('Please be advised that astroNN mag_to_absmag expects parallax in (arcsecond)')
    return mag + 5 * (np.log10(parallax) + 1)


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
    return (1 / (10 ** (((absmag - mag) / 5) - 1))) * u.parsec


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
    return 5 * np.log10(fakemag) - 2


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
    return 10 ** (0.2 * absmag + 2)


def fakemag_to_pc(fakemag, mag):
    """
    NAME:
        fakemag_to_absmag
    PURPOSE:
        To convert fakemag to parsec
    INPUT:
        fakemag (float, ndarray): fakemag
        mag (float, ndarray): magnitude
    OUTPUT:
        parsec (float, ndarray with astropy unit) in pc
    HISTORY:
        2018-Jan-31 - Written - Henry Leung (University of Toronto)
    """
    return absmag_to_pc(5 * np.log10(fakemag) - 2, mag) * u.parsec
