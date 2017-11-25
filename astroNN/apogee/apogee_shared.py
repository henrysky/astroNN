# ---------------------------------------------------------#
#   astroNN.apogee.apogee_shared: shared functions for apogee
# ---------------------------------------------------------#

import os


def apogee_env():
    """
    NAME: apogee_env
    PURPOSE: get APOGEE enviroment variable
    INPUT:
    OUTPUT: path
    """
    _APOGEE = os.getenv('SDSS_LOCAL_SAS_MIRROR')
    if _APOGEE is None:
        raise RuntimeError("Cannot find APOGEE enviroment variable SDSS_LOCAL_SAS_MIRROR")
    return _APOGEE


def apogee_default_dr(dr=None):
    """
    NAME: apogee_default_dr
    PURPOSE: Check if dr arguement is provided, if none then use default
    INPUT: dr
    OUTPUT: dr
    """
    if dr is None:
        dr = 14
        print('dr is not provided, using default dr={}'.format(dr))
    else:
        pass
    return dr