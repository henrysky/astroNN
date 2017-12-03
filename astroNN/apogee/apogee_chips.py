# ---------------------------------------------------------#
#   astroNN.apogee.apogee_chips: tools for dealing with apogee camera chips
# ---------------------------------------------------------#

import numpy as np

from astroNN.apogee.apogee_shared import apogee_default_dr


def chips_pix_info(dr=None):
    """
    NAME: chips_pix_info
    PURPOSE: return chips info according to dr
    INPUT:
        dr = 13 or 14
    OUTPUT: chips info
    HISTORY:
        2017-Nov-27 Henry Leung
    """
    dr = apogee_default_dr(dr=dr)

    if dr == 13:
        blue_start = 322
        blue_end = 3243
        green_start = 3648
        green_end = 6049
        red_start = 6412
        red_end = 8306
    elif dr == 14:
        blue_start = 246
        blue_end = 3274
        green_start = 3585
        green_end = 6080
        red_start = 6344
        red_end = 8335
    else:
        raise ValueError('Only DR13 and DR14 are supported')

    return [blue_start, blue_end, green_start, green_end, red_start, red_end]


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
    info = chips_pix_info(dr=dr)

    arr1 = np.arange(0, info[0], 1) # Blue chip gap
    arr2 = np.arange(info[1], info[2], 1) # Green chip gap
    arr3 = np.arange(info[3], info[4], 1) # Red chip gap
    arr4 = np.arange(info[5], len(single_spec), 1)
    single_spec = np.delete(single_spec, arr4)
    single_spec = np.delete(single_spec, arr3)
    single_spec = np.delete(single_spec, arr2)
    single_spec = np.delete(single_spec, arr1)

    return single_spec


def wavelegnth_solution(dr=None):
    """
    NAME: wavelegnth_solution
    PURPOSE: to return wavelegnth_solution
    INPUT:
        dr = 13 or 14
    OUTPUT: 3 wavelength solution array
    HISTORY:
        2017-Nov-20 Henry Leung
        apStarWavegrid was provided by Jo Bovy's apogee tools (Toronto)
    """
    dr = apogee_default_dr(dr=dr)
    info = chips_pix_info(dr=dr)
    apStarWavegrid = 10.**np.arange(4.179, 4.179+8575*6.*10.**-6., 6. * 10. ** -6.)

    lambda_blue = apStarWavegrid[info[0]:info[1]]
    lambda_green = apStarWavegrid[info[2]:info[3]]
    lambda_red = apStarWavegrid[info[4]:info[5]]

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
    info = chips_pix_info(dr=dr)

    blue = info[1]-info[0]
    green = info[3]-info[2]
    red = info[5]-info[4]

    lambda_blue = spec[0:blue]
    lambda_green = spec[blue:blue+green]
    lambda_red = spec[blue+green:blue+green+red]

    return lambda_blue, lambda_green, lambda_red


def cont_normalization(dispersion, fluxes, flux_var, continuum_mask, degree=2):
    """
    NAME: cont_normalization
    PURPOSE:
        Fit Chebyshev polynomials to the flux values in the continuum mask. Fluxes
        from multiple stars can be given, and the resulting continuum will have the
        same shape as `fluxes`.
    INPUT:
        dispersion: The dispersion
        fluxes: the spectra
        flux_var: the spectra uncertainty
        continuum_mask: A mask for continuum pixels to use
        degree: The degree of Chebyshev polynomial to use in each region.
    OUTPUT: 3 arrays
    HISTORY:
        2017-Nov-20 Henry Leung
    """

    region_masks = [np.ones_like(dispersion, dtype=bool)]

    dispersion = np.array(dispersion).flatten()
    fluxes = np.atleast_2d(fluxes)
    flux_uncertainties = np.atleast_2d(flux_var)

    N_stars = fluxes.shape[0]
    assert fluxes.shape[1] == dispersion.size

    i_variances = 1.0 / flux_uncertainties ** 2

    # Use only continuum pixels.
    i_variances[:, ~continuum_mask] += 200 ** 2
    # i_variances[~np.isfinite(i_variances)] = 0

    continuum = np.ones_like(fluxes, dtype=float)
    for i, (flux, i_var) in enumerate(zip(fluxes, i_variances)):
        for region_mask in region_masks:
            fitted_mask = region_mask * continuum_mask
            f = np.polynomial.chebyshev.Chebyshev.fit(
                x=dispersion[fitted_mask], y=flux[fitted_mask],
                w=i_var[fitted_mask], deg=degree)

            continuum[i, region_mask] *= f(dispersion[region_mask])

    return continuum
