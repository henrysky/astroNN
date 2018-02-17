# ---------------------------------------------------------#
#   astroNN.apogee.chips: tools for dealing with apogee camera chips
# ---------------------------------------------------------#

import os

import numpy as np

import astroNN
from astroNN.apogee.apogee_shared import apogee_default_dr


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
           # info[0] refers to the location where blue chips starts
           # info[1] refers to the location where blue chips ends
           # info[2] refers to the location where green chips starts
           # info[3] refers to the location where blue chips end
           # info[4] refers to the location where red chips starts
           # info[5] refers to the location where red chips ends
           # info[6] refers to the total number of pixels after deleting gap
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
        spectra (ndarray): The original 8575 pixels spectrum/spectra
        dr (int): APOGEE DR, example dr=14
    OUTPUT:
        spectra (ndarray): Gap deleted spectrum/spectra
    HISTORY:
        2017-Oct-26 - Written - Henry Leung (University of Toronto)
        2017-Dec-16 - Update - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)
    spectra = np.atleast_2d(spectra)

    if spectra.shape[1] != 8575:
        raise EnvironmentError('Are you sure you are giving astroNN APOGEE spectra?')

    info = chips_pix_info(dr=dr)

    spectra = spectra[:, np.r_[info[0]:info[1], info[2]:info[3], info[4]:info[5]]]

    return spectra


def wavelength_solution(dr=None):
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
           # lambda_blue refers to the wavelength solution for each pixel in blue chips
           # lambda_green refers to the wavelength solution for each pixel in green chips
           # lambda_red refers to the wavelength solution for each pixel in red chips
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
        split a single gap deleted spectra into RGB chips
        will delete the gap if detected
    INPUT:
        dr (int): APOGEE DR, example dr=14
    OUTPUT:
        (ndarray): An array from blue, green, red chips
    HISTORY:
        2017-Nov-20 - Written - Henry Leung (University of Toronto)
        2017-Dec-17 - Update - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)
    info = chips_pix_info(dr=dr)
    blue = info[1] - info[0]
    green = info[3] - info[2]
    red = info[5] - info[4]

    spectra = np.atleast_2d(spectra)

    if spectra.shape[1] == 8575:
        spectra = gap_delete(spectra, dr=dr)
    if spectra.shape[1] != 8575 and spectra.shape[1] > 8500:
        raise EnvironmentError('Are you sure you are giving astroNN APOGEE spectra?')

    spectra_blue = spectra[:, 0:blue]
    spectra_green = spectra[:, blue:(blue + green)]
    spectra_red = spectra[:, (blue + green):(blue + green + red)]

    return spectra_blue, spectra_green, spectra_red


def bitmask_boolean(bitmask, target_bit):
    """
    NAME:
        bitmask_boolean
    PURPOSE:
        Turn bitmask to boolean with provided bitmask array and target bit to mask
    INPUT:
        bitmask (ndaray): bitmask
        target_bit (list): target bit to mask
    OUTPUT:
        (ndarray, boolean): boolean array, True for clean, False for masked
    HISTORY:
        2018-Feb-03 - Written - Henry Leung (University of Toronto)
    """
    target_bit = np.asarray(target_bit)
    target_bit = np.sum(2 ** target_bit)
    bitmask = np.atleast_2d(bitmask)
    boolean_output = np.ones(bitmask.shape, dtype=bool)
    boolean_output[(bitmask & target_bit) != 0] = False
    return boolean_output


def bitmask_decompositor(bitmask):
    """
    NAME:
        bitmask_decompositor
    PURPOSE:
        To decompose a bit from bitmask array to individual bit
    INPUT:
        bitmask (ndaray): bitmask
    OUTPUT:
        (ndarray, boolean): boolean array, True for clean, False for masked
    HISTORY:
        2018-Feb-04 - Written - Henry Leung (University of Toronto)
    """
    bitmask_num = np.array(bitmask)
    if bitmask_num < 0:
        raise ValueError("Your number is not valid, this value must not from a bitmask")
    if bitmask_num == 0:
        print('0 corresponds to good pixel, thus this bit cannot be decomposed')
        return None
    decomposited_bits = [int(np.log2(bitmask_num))]
    while True:
        if bitmask_num - 2 ** decomposited_bits[-1] == 0:
            decomposited_bits.sort()
            decomposited_bits = np.array(decomposited_bits)
            break
        bitmask_num -= 2 ** decomposited_bits[-1]
        if bitmask_num != 0:
            decomposited_bits.append(int(np.log2(bitmask_num)))

    return decomposited_bits


def continuum(spectra, spectra_err, cont_mask=None, deg=2, dr=None):
    """
    NAME:
        continuum
    PURPOSE:
        Fit Chebyshev polynomials to the flux values in the continuum mask by chips. The resulting continuum will have
        the same shape as `fluxes`.
    INPUT:
        spectra (ndaray): spectra
        spectra_err (ndaray): spectra uncertainty (std/sigma)
        cont_mask (ndaray): A mask for continuum pixels to use, or not specifying it to use mine
        deg (int): The degree of Chebyshev polynomial to use in each region, default is 2 which works the best so far
        dr (int): APOGEE DR, example dr=14
    OUTPUT:
        (ndarray): normalized flux
        (ndarray): normalized error flux
    HISTORY:
        2017-Dec-04 - Written - Henry Leung (University of Toronto)
        2017-Dec-16 - Update - Henry Leung (University of Toronto)
    """

    dr = apogee_default_dr(dr=dr)
    info = chips_pix_info(dr=dr)

    spectra = np.atleast_2d(spectra)
    flux_errs = np.atleast_2d(spectra_err)
    flux_vars = np.square(flux_errs)

    if spectra.shape[1] == 8575:
        spectra = gap_delete(spectra, dr=dr)
    if spectra.shape[1] != 8575 and spectra.shape[1] > 8500:
        raise EnvironmentError('Are You Sure you are giving astroNN APOGEE spectra?')

    yivars = 1 / flux_vars  # Inverse variance weighting
    pix = np.arange(info[6])  # Array with size gap_deleted spectra
    cont_arr = np.zeros(spectra.shape)  # Corrected spectra
    cont_arr_err = np.zeros(spectra.shape)  # Corrected error spectra

    spectra_blue, spectra_green, spectra_red = chips_split(spectra, dr=dr)
    yivars_blue, yivars_green, yivars_red = chips_split(yivars, dr=dr)
    yerrs_blue, yerrs_green, yerrs_red = chips_split(flux_errs, dr=dr)

    pix_blue, pix_green, pix_red = chips_split(pix, dr=dr)
    pix_blue, pix_green, pix_red = pix_blue[0], pix_green[0], pix_red[0]

    if cont_mask is None:
        dir = os.path.join(os.path.dirname(astroNN.__path__[0]), 'astroNN', 'data', 'dr{}_contmask.npy'.format(dr))
        cont_mask = np.load(dir)

    con_mask_blue, cont_mask_green, con_mask_red = chips_split(cont_mask, dr=dr)
    con_mask_blue, cont_mask_green, con_mask_red = con_mask_blue[0], cont_mask_green[0], con_mask_red[0]

    blue = info[1] - info[0]
    green = info[3] - info[2]
    red = info[5] - info[4]

    masked_blue = np.arange(blue)[con_mask_blue]
    masked_green = np.arange(green)[cont_mask_green]
    masked_red = np.arange(red)[con_mask_red]

    for counter, (spectrum_blue, spectrum_green, spectrum_red, yivar_blue, yivar_green, yivar_red, yerr_blue,
                  yerr_green, yerr_red) in enumerate(zip(spectra_blue, spectra_green, spectra_red, yivars_blue,
                                                         yivars_green, yivars_red, yerrs_blue, yerrs_green, yerrs_red)):
        ###############################################################
        fit = np.polynomial.chebyshev.Chebyshev.fit(x=masked_blue, y=spectrum_blue[con_mask_blue],
                                                    w=yivar_blue[con_mask_blue], deg=deg)

        for local_counter, element in enumerate(pix_blue):
            cont_arr[counter, element] = spectrum_blue[local_counter] / fit(local_counter)
            # We want std/simga not ivar, also need to deal with normalize**2
            cont_arr_err[counter, element] = yerr_blue[local_counter] / (fit(local_counter) ** 2)

        ###############################################################
        fit = np.polynomial.chebyshev.Chebyshev.fit(x=masked_green, y=spectrum_green[cont_mask_green]
                                                    , w=yivar_green[cont_mask_green], deg=deg)

        for local_counter, element in enumerate(pix_green):
            cont_arr[counter, element] = spectrum_green[local_counter] / fit(local_counter)
            # We want std/simga not ivar
            cont_arr_err[counter, element] = yerr_green[local_counter] / (fit(local_counter) ** 2)

        ###############################################################
        fit = np.polynomial.chebyshev.Chebyshev.fit(x=masked_red, y=spectrum_red[con_mask_red],
                                                    w=yivar_red[con_mask_red], deg=deg)

        for local_counter, element in enumerate(pix_red):
            cont_arr[counter, element] = spectrum_red[local_counter] / fit(local_counter)
            # We want std/simga not ivar
            cont_arr_err[counter, element] = yerr_red[local_counter] / (fit(local_counter) ** 2)

        ###############################################################

    # Clip all bad errors (they are useless anyway)
    cont_arr_err[cont_arr_err > 2] = 0
    cont_arr_err[cont_arr_err < 0] = 0

    return cont_arr, cont_arr_err
