# ---------------------------------------------------------#
#   astroNN.apogee.apogee_chips: tools for dealing with apogee camera chips
# ---------------------------------------------------------#

import numpy as np

from astroNN.apogee.apogee_shared import apogee_default_dr


def gap_delete(single_spec, dr=None):
    """
    NAME: gap_delete
    PURPOSE: delete the gap between APOGEE camera
    INPUT:
        single_spec = single spectra array
        dr = 13 or 14
    OUTPUT: corrected array
    HISTORY:
        2017-Oct-26 Henry Leung
    """
    dr = apogee_default_dr(dr=dr)

    if dr == 13:
        arr1 = np.arange(0, 322, 1) # Blue chip gap
        arr2 = np.arange(3243, 3648, 1) # Green chip gap
        arr3 = np.arange(6049, 6412, 1) # Red chip gap
        arr4 = np.arange(8306, len(single_spec), 1)
        single_spec = np.delete(single_spec, arr4)
        single_spec = np.delete(single_spec, arr3)
        single_spec = np.delete(single_spec, arr2)
        single_spec = np.delete(single_spec, arr1)
    elif dr == 14:
        arr1 = np.arange(0, 246, 1) # Blue chip gap
        arr2 = np.arange(3274, 3585, 1) # Green chip gap
        arr3 = np.arange(6080, 6344, 1) # Red chip gap
        arr4 = np.arange(8335, len(single_spec), 1)
        single_spec = np.delete(single_spec, arr4)
        single_spec = np.delete(single_spec, arr3)
        single_spec = np.delete(single_spec, arr2)
        single_spec = np.delete(single_spec, arr1)
    else:
        raise ValueError('Only DR13 and DR14 are supported')

    return single_spec


def wavelegnth_solution(dr=None):
    """
    NAME: wavelegnth_solution
    PURPOSE: to return wavelegnth_solution
    INPUT:
        dr = 13 or 14
    OUTPUT: 3 wavelegnth solution array
    HISTORY:
        2017-Nov-20 Henry Leung
    """
    dr = apogee_default_dr(dr=dr)
    lambda_blue = np.zeros(3028)
    lambda_green = np.zeros(2495)
    lambda_red = np.zeros(1991)

    lambda_blue[0] = 15152.211
    lambda_green[0] = 15867.555
    lambda_red[0] = 16484.053

    dispersion_10_ratio = 10 ** (6e-6)

    if dr == 14:
        for i in range(1, 3028):
            lambda_blue[i] = lambda_blue[i-1] * dispersion_10_ratio

        for i in range(1, 2495):
            lambda_green[i] = lambda_green[i-1] * dispersion_10_ratio

        for i in range(1, 1991):
            lambda_red[i] = lambda_red[i-1] * dispersion_10_ratio

        # lambda_blue = np.linspace(15146, 15910, 3028, endpoint=True)
        # lambda_green = np.linspace(15961, 16434, 2495, endpoint=True)
        # lambda_red = np.linspace(16476, 16953, 1991, endpoint=True)
    else:
        raise ValueError('Only DR14 are supported')

    return lambda_blue, lambda_green, lambda_red


def chips_split(spec, dr=None):
    """
    NAME: chips_split
    PURPOSE: split a single spectra into RGB chips
    INPUT:
        dr = 13 or 14
    OUTPUT: 3 arrays
    HISTORY:
        2017-Nov-20 Henry Leung
    """
    dr = apogee_default_dr(dr=dr)

    if dr == 14:
        lambda_blue = spec[0:3028]
        lambda_green = spec[3028:5523]
        lambda_red = spec[5523:]
    else:
        raise ValueError('Only DR14 are supported')

    return lambda_blue, lambda_green, lambda_red