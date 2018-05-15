from astroNN.lamost.lamost_shared import lamost_default_dr
import numpy as np


def wavelength_solution(dr=None):
    """
    To return wavelegnth_solution

    :param dr: data release
    :type dr: Union(int, NoneType)
    :return: wavelength solution array
    :rtype: ndarray
    :History: 2018-Mar-15 - Written - Henry Leung (University of Toronto)
    """
    dr = lamost_default_dr(dr=dr)

    apstar_wavegrid = 10. ** np.arange(3.5682, 3.5682 + 3909 * 1. * 10. ** -4., 4. * 10. ** -4.)

    return apstar_wavegrid
