# ---------------------------------------------------------#
#   astroNN.gaia.gaia_shared: shared functions for apogee
# ---------------------------------------------------------#

import os
import numpy as np


def gaia_env():
    """
    NAME: gaia_env
    PURPOSE: get GAIA enviroment variable
    INPUT:
    OUTPUT: path
    """
    _GAIA = os.getenv('GAIA_TOOLS_DATA')
    if _GAIA is None:
        raise RuntimeError("Gaia enviroment variable GAIA_TOOLS_DATA not set")
    return _GAIA


def gaia_default_dr(dr=None):
    """
    NAME: gaia_default_dr
    PURPOSE: Check if dr arguement is provided, if none then use default
    INPUT: dr
    OUTPUT: dr
    """
    if dr is None:
        dr = 1
        print('dr is not provided, using default dr={}'.format(dr))
    else:
        pass
    return dr


def mag_to_absmag(mag, parallax):
    """
    NAME: mag_to_absmag
    PURPOSE: To convert appearant magnitude to absolute magnitude
    INPUT:
    OUTPUT: the model
    HISTORY:
        2017-Oct-14 Henry Leung
    """
    return mag + 5 * (np.log10(parallax) + 1)


def absmag_to_pc(absmag, mag):
    """
    NAME: absmag_to_pc
    PURPOSE: To convert absolute magnitude to parsec
    INPUT:
    OUTPUT: the model
    HISTORY:
        2017-Nov-16 Henry Leung
    """
    return 1 / (10 ** (((absmag - mag) / 5)- 1))