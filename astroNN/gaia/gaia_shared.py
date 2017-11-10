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
    return os.getenv('GAIA_TOOLS_DATA')


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


def to_absmag(mag, parallax):
    """
    NAME: to_absmag
    PURPOSE: To convert appearant magnitude to absolute magnitude
    INPUT:
    OUTPUT: the model
    HISTORY:
        2017-Oct-14 Henry Leung
    """
    return mag + 5 * (np.log10(parallax / 1000) + 1)
