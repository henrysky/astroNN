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
    lamost_default_dr(dr=dr)

    # delibreately add 1e-5 to prevent numpy to generate an extra element
    lamost_wavegrid = 10. ** np.arange(3.5682, 3.5682 - 1e-5 + 3909 * 10. ** -4., 10. ** -4.)

    return lamost_wavegrid
