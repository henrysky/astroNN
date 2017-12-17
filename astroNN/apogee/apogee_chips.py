# ---------------------------------------------------------#
#   astroNN.apogee.apogee_chips: tools for dealing with apogee camera chips
# ---------------------------------------------------------#

import numpy as np
import os

from astroNN.apogee.apogee_shared import apogee_default_dr
import astroNN


def chips_pix_info(dr=None):
    """
    NAME:
        chips_pix_info
    PURPOSE:
        return chips info according to dr
    INPUT:
        dr (int): APOGEE DR, example dr=14
    OUTPUT:
        (list): The starting and ending pixels location of APOGEE camera chips in the original 8000s pixels spectra
    HISTORY:
        2017-Nov-27 - Written - Henry Leung (University of Toronto)
        2017-Dec-16 - Update - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)

    if dr == 14 or dr == 13:
        blue_start = 246
        blue_end = 3274
        green_start = 3585
        green_end = 6080
        red_start = 6344
        red_end = 8335
        total_pixel = 7514
    else:
        raise ValueError('Only DR13 and DR14 are supported')

    return [blue_start, blue_end, green_start, green_end, red_start, red_end, total_pixel]


def gap_delete(spectra, dr=None):
    """
    NAME:
        gap_delete
    PURPOSE:
        delete the gap between APOGEE CCDs from the original 8000s pixels spectra
    INPUT:
        spectra (ndarray): The original 8000s pixels spectrum/spectra
        dr (int): APOGEE DR, example dr=14
    OUTPUT:
        spectra (ndarray): Gap deleted spectrum/spectra
    HISTORY:
        2017-Oct-26 - Written - Henry Leung (University of Toronto)
        2017-Dec-16 - Update - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)
    info = chips_pix_info(dr=dr)

    spectra = np.atleast_2d(spectra)
    spectra = spectra[:, np.r_[info[0]:info[1], info[2]:info[3], info[4]:info[5]]]

    return spectra


def wavelegnth_solution(dr=None):
    """
    NAME:
        wavelegnth_solution
    PURPOSE:
        to return wavelegnth_solution
        apStarWavegrid was provided by Jo Bovy's apogee tools (Toronto)
    INPUT:
        dr (int): APOGEE DR, example dr=14
    OUTPUT:
        (ndarray): 3 wavelength solution array
    HISTORY:
        2017-Nov-20 - Written - Henry Leung (University of Toronto)
        2017-Dec-16 - Update - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)
    info = chips_pix_info(dr=dr)

    apStarWavegrid = 10. ** np.arange(4.179, 4.179 + 8575 * 6. * 10. ** -6., 6. * 10. ** -6.)

    lambda_blue = apStarWavegrid[info[0]:info[1]]
    lambda_green = apStarWavegrid[info[2]:info[3]]
    lambda_red = apStarWavegrid[info[4]:info[5]]

    return lambda_blue, lambda_green, lambda_red


def chips_split(spectra, dr=None):
    """
    NAME:
        chips_split
    PURPOSE:
        split a single spectra into RGB chips
    INPUT:
        dr (int): APOGEE DR, example dr=14
    OUTPUT:
        (ndarray): 3 array from blue, green, red chips
    HISTORY:
        2017-Nov-20 - Written - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)
    info = chips_pix_info(dr=dr)

    blue = info[1] - info[0]
    green = info[3] - info[2]
    red = info[5] - info[4]

    spectra_blue = spectra[0:blue]
    spectra_green = spectra[blue:(blue + green)]
    spectra_red = spectra[(blue + green):(blue + green + red)]

    return spectra_blue, spectra_green, spectra_red


def continuum(fluxes, flux_vars, cont_mask=None, deg=2, dr=None):
    """
    NAME:
        continuum
    PURPOSE:
        Fit Chebyshev polynomials to the flux values in the continuum mask by chips. The resulting continuum will have
        the same shape as `fluxes`.
    INPUT:
        fluxes (ndaray): the spectra without gap, run astroNN.apogee.apogee_chips.gap_delete fisrt
        flux_vars (ndaray): the spectra uncertainty
        cont_mask (ndaray): A mask for continuum pixels to use, or not specifying it to use mine
        deg (int): The degree of Chebyshev polynomial to use in each region, default is 2 which works the best so far
        dr (int): APOGEE DR, example dr=14
    OUTPUT:
        (ndarray): normalized flux
    HISTORY:
        2017-Dec-04 - Written - Henry Leung (University of Toronto)
        2017-Dec-16 - Update - Henry Leung (University of Toronto)
    """

    dr = apogee_default_dr(dr=dr)
    info = chips_pix_info(dr=dr)

    fluxes = np.atleast_2d(fluxes)
    flux_vars = np.atleast_2d(flux_vars)

    yivars = 1 / flux_vars  # Inverse variance weighting
    pix = np.arange(info[6])  # Array with size gap_deleted spectra

    cont_arr = np.zeros(fluxes.shape)

    if cont_mask is None:
        dir = os.path.join(os.path.dirname(astroNN.__path__[0]), 'astroNN', 'data', 'dr{}_contmask.npy'.format(dr))
        cont_mask = np.load(dir)

    blue_mask, green_mask, red_mask = chips_split(cont_mask, dr=dr)

    blue = info[1] - info[0]
    green = info[3] - info[2]
    red = info[5] - info[4]

    for counter, (spectrum, yivar) in enumerate(zip(fluxes, yivars)):

        blue_spec, green_spec, red_spec = chips_split(spectrum, dr=dr)
        blue_pix, green_pix, red_pix = chips_split(pix, dr=dr)
        blue_err, green_err, red_err = chips_split(yivar, dr=dr)

        ###############################################################
        fit = np.polynomial.chebyshev.Chebyshev.fit(x=np.arange(blue)[blue_mask], y=blue_spec[blue_mask], w=blue_err[blue_mask],
                                                    deg=deg)

        for local_counter, element in enumerate(blue_pix):
            cont_arr[counter, element] = blue_spec[local_counter] / fit(local_counter)

        ###############################################################
        fit = np.polynomial.chebyshev.Chebyshev.fit(x=np.arange(green)[green_mask], y=green_spec[green_mask], w=green_err[green_mask],
                                                    deg=deg)

        for local_counter, element in enumerate(green_pix):
            cont_arr[counter, element] = green_spec[local_counter] / fit(local_counter)

        ###############################################################
        fit = np.polynomial.chebyshev.Chebyshev.fit(x=np.arange(red)[red_mask], y=red_spec[red_mask], w=red_err[red_mask],
                                                    deg=deg)

        for local_counter, element in enumerate(red_pix):
            cont_arr[counter, element] = red_spec[local_counter] / fit(local_counter)
        ###############################################################

    return cont_arr
