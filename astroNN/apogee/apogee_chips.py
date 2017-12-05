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


def continuum(fluxes, flux_vars, cont_mask, deg=2, dr=None):
    """
    NAME: continuum
    PURPOSE:
        Fit Chebyshev polynomials to the flux values in the continuum mask by chips. The resulting continuum will have the
        same shape as `fluxes`.
    INPUT:
        fluxes: the spectra without gap, run astroNN.apogee.apogee_chips.gap_delete fisrt
        flux_vars: the spectra uncertainty
        cont_mask: A mask for continuum pixels to use
        deg: The degree of Chebyshev polynomial to use in each region.
    OUTPUT: normalized flux
    HISTORY:
        2017-Dec-04 Henry Leung
    """

    dr = apogee_default_dr(dr=dr)
    info = chips_pix_info(dr=dr)

    blue_spec, green_spec, red_spec = chips_split(fluxes, dr=dr)
    blue_err, green_err, red_err = chips_split(flux_vars, dr=dr)
    blue_mask, green_mask, red_mask = chips_split(cont_mask, dr=dr)

    cont_arr = np.zeros(fluxes.shape)

    blue = info[1]-info[0]
    green = info[3]-info[2]
    red = info[5]-info[4]

    ###############################################################
    flux = blue_spec
    flux_var = blue_err
    pix = np.arange(blue)
    y = flux[blue_mask]
    x = pix[blue_mask]
    yivar = 1.0 / ((flux_var[blue_mask]) ** 2)
    yivar += 200**2
    fit = np.polynomial.chebyshev.Chebyshev.fit(x=x, y=y, w=yivar, deg=deg)

    for element in pix:
        cont_arr[element] = blue_spec[element] / fit(element)

    ###############################################################
    flux = green_spec
    flux_var = green_err
    pix = np.arange(green)
    y = flux[green_mask]
    x = pix[green_mask]
    yivar = 1.0 / ((flux_var[green_mask]) ** 2)
    yivar += 200**2
    fit = np.polynomial.chebyshev.Chebyshev.fit(x=x, y=y, w=yivar, deg=deg)

    for element in pix:
        cont_arr[element + blue] = green_spec[element] / fit(element)

    ###############################################################
    flux = red_spec
    flux_var = red_err
    pix = np.arange(red)
    y = flux[red_mask]
    x = pix[red_mask]
    yivar = 1.0 / ((flux_var[red_mask]) ** 2)
    yivar += 200**2
    fit = np.polynomial.chebyshev.Chebyshev.fit(x=x, y=y, w=yivar, deg=deg)

    for element in pix:
        cont_arr[element + blue + green] = red_spec[element] / fit(element)

    return cont_arr
