import numpy as np

from astroNN.lamost.lamost_shared import lamost_default_dr


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
    lamost_wavegrid = 10.0 ** np.arange(
        3.5682, 3.5682 - 1e-5 + 3909 * 10.0**-4.0, 10.0**-4.0
    )

    return lamost_wavegrid


def smooth_spec(flux, ivar, wavelength, L=50):
    """
    Smooth a spectrum with a running Gaussian.

    :param flux: The observed flux array.
    :type flux: ndarray
    :param ivar: The inverse variances of the fluxes.
    :type ivar: ndarray
    :param wavelength: An array of the wavelengths.
    :type wavelength: ndarray
    :param L: The width of the Gaussian in pixels.
    :type L: int
    :returns: An array of smoothed fluxes
    :rtype: ndarray
    """

    # Partial Credit: https://github.com/chanconrad/slomp/blob/master/lamost.py
    w = np.exp(-0.5 * (wavelength[:, None] - wavelength[None, :]) ** 2 / L**2)
    denominator = np.dot(ivar, w.T)
    numerator = np.dot(flux * ivar, w.T)
    bad_pixel = denominator == 0
    smoothed = np.zeros(numerator.shape)
    smoothed[~bad_pixel] = numerator[~bad_pixel] / denominator[~bad_pixel]
    return smoothed


def pseudo_continuum(flux, ivar, wavelength=None, L=50, dr=None):
    """
    Pseudo-Continuum normalise a spectrum by dividing by a Gaussian-weighted smoothed spectrum.

    :param flux: The observed flux array.
    :type flux: ndarray
    :param ivar: The inverse variances of the fluxes.
    :type ivar: ndarray
    :param wavelength: An array of the wavelengths.
    :type wavelength: ndarray
    :param L: [optional] The width of the Gaussian in pixels.
    :type L: int
    :param dr: [optional] dara release
    :type dr: int
    :returns: Continuum normalized flux and flux uncerteinty
    :rtype: ndarray
    """

    # Partial Credit: https://github.com/chanconrad/slomp/blob/master/lamost.py
    if dr is None:
        dr = lamost_default_dr(dr)

    if wavelength is None:
        wavelength = wavelength_solution(dr=dr)

    smoothed_spec = smooth_spec(wavelength, flux, ivar, L)
    norm_flux = flux / smoothed_spec
    norm_ivar = smoothed_spec * ivar * smoothed_spec

    bad_pixel = ~np.isfinite(norm_flux)
    norm_flux[bad_pixel] = 1.0
    norm_ivar[bad_pixel] = 0.0

    return norm_flux, norm_ivar
