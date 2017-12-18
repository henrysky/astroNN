# ---------------------------------------------------------#
#   astroNN.gaia.gaia_shared: shared functions for apogee
# ---------------------------------------------------------#

import os
import numpy as np
from astropy.io import fits

from  astroNN.gaia.downloader import tgas


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
        raise RuntimeError("Gaia enviroment variable GAIA_TOOLS_DATA not set")
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


def tgas_load(dr=None):
    """
    NAME:
        tgas_load
    PURPOSE:
        to load useful parameters from multiple TGAS files
    INPUT:
        dr (int): GAIA DR, example dr=1
    OUTPUT:
    HISTORY:
        2017-Dec-17 - Written - Henry Leung (University of Toronto)
    """
    dr = gaia_default_dr(dr=dr)
    tgas_list = tgas(dr=dr)

    ra_gaia = np.array([])
    dec_gaia = np.array([])
    pmra_gaia = np.array([])
    pmdec_gaia = np.array([])
    parallax_gaia = np.array([])
    parallax_error_gaia = np.array([])
    g_band_gaia = np.array([])

    for i in tgas_list:
        gaia = fits.open(i)
        ra_gaia = np.concatenate((ra_gaia, gaia[1].data['RA']))
        dec_gaia = np.concatenate((dec_gaia, gaia[1].data['DEC']))
        pmra_gaia = np.concatenate((pmra_gaia, gaia[1].data['PMRA']))
        pmdec_gaia = np.concatenate((pmdec_gaia, gaia[1].data['PMDEC']))
        parallax_gaia = np.concatenate((parallax_gaia, gaia[1].data['parallax']))
        parallax_error_gaia = np.concatenate((parallax_error_gaia, gaia[1].data['parallax_error']))
        g_band_gaia = np.concatenate((g_band_gaia, gaia[1].data['phot_g_mean_mag']))
        gaia.close()

    return ra_gaia, dec_gaia, pmra_gaia, pmdec_gaia, parallax_gaia, parallax_error_gaia, g_band_gaia


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
        parallax (float)
    HISTORY:
        2017-Nov-16 - Written - Henry Leung (University of Toronto)
    """
    return 1 / (10 ** (((absmag - mag) / 5)- 1))