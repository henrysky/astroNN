# ---------------------------------------------------------#
#   astroNN.apogee.apogee_shared: shared functions for apogee
# ---------------------------------------------------------#

import os


def apogee_env():
    """
    Get APOGEE environment variable

    :return: path to APOGEE local dir
    :rtype: str
    :History: 2017-Oct-26 - Written - Henry Leung (University of Toronto)
    """
    from astroNN.config import ENVVAR_WARN_FLAG

    _APOGEE = os.getenv("SDSS_LOCAL_SAS_MIRROR")
    if _APOGEE is None and ENVVAR_WARN_FLAG is True:
        print("WARNING! APOGEE environment variable SDSS_LOCAL_SAS_MIRROR not set")

    return _APOGEE


def apogee_default_dr(dr=None):
    """
    Check if dr argument is provided, if none then use default

    :param dr: APOGEE DR
    :type dr: int
    :return: APOGEE DR
    :rtype: int
    :History:
        | 2017-Oct-26 - Written - Henry Leung (University of Toronto)
        | 2018-Sept-08 - Updated - Henry Leung (University of Toronto)
    """
    if dr == 15:
        print(
            "SDSS APOGEE DR15 is equivalent to DR14, so astroNN is using DR14 even you set DR15"
        )
        dr = 14

    if dr is None:
        try:
            redux_ver = os.environ[
                "RESULTS_VERS"
            ]  # RESULTS_VERS is from Jo Bovy APOGEE Tool
            if redux_ver == "v402":
                dr = 11
            elif redux_ver == "v603":
                dr = 12
            elif redux_ver == "l30e.2":
                dr = 13
            elif redux_ver == "l31c.2":
                dr = 14
            elif redux_ver == "l33":
                dr = 16
            elif redux_ver == "dr17":
                dr = 17
        except KeyError:
            pass

        if dr is None:  # if it is still None
            dr = 17
            print(f"dr is not provided, using default dr={dr}")
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
    import numpy as np

    if isinstance(arr, np.ndarray) or isinstance(arr, list):
        arr_copy = np.array(arr)  # make a copy
        for i in range(arr_copy.shape[0]):
            arr_copy[i] = str("".join(filter(str.isdigit, arr_copy[i])))
        return arr_copy
    else:
        return str("".join(filter(str.isdigit, arr)))
