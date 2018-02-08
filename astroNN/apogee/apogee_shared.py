# ---------------------------------------------------------#
#   astroNN.apogee.apogee_shared: shared functions for apogee
# ---------------------------------------------------------#

import os


def apogee_env():
    """
    NAME:
        apogee_env
    PURPOSE:
        get APOGEE enviroment variable
    INPUT:
        None
    OUTPUT:
        (path)
    HISTORY:
        2017-Oct-26 - Written - Henry Leung
    """
    _APOGEE = os.getenv('SDSS_LOCAL_SAS_MIRROR')
    if _APOGEE is None:
        print("WARNING! APOGEE environment variable SDSS_LOCAL_SAS_MIRROR not set")

    return _APOGEE


def apogee_default_dr(dr=None):
    """
    NAME:
        apogee_default_dr
    PURPOSE:
        Check if dr arguement is provided, if none then use default
    INPUT:
        dr (int): APOGEE DR, example dr=14
    OUTPUT:
        dr (int): APOGEE DR, example dr=14
    HISTORY:
        2017-Oct-26 - Written - Henry Leung (University of Toronto)
    """
    if dr is None:
        dr = 14
        print('dr is not provided, using default dr={}'.format(dr))
    else:
        pass

    return dr


def apogeeid_digit(arr):
    """
    NAME:
        apogeeid_digit
    PURPOSE:
        Extract digits from apogeeid because its too painful to deal with APOGEE ID in h5py
    INPUT:
        arr (ndarray): apogee_id
    OUTPUT:
        apogee_id with digits only (ndarray)
    HISTORY:
        2017-Oct-26 - Written - Henry Leung (University of Toronto)
    """

    return str(''.join(filter(str.isdigit, arr)))
