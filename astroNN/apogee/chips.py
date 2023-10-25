# ---------------------------------------------------------#
#   astroNN.apogee.chips: tools for dealing with apogee camera chips
# ---------------------------------------------------------#

import os
import warnings

import numpy as np

import astroNN
import astroNN.data
from astroNN.apogee.apogee_shared import apogee_default_dr


def chips_pix_info(dr=None):
    """
    To return chips info according to dr

    :param dr: data release
    :type dr: Union(int, NoneType)
    :return:
        | The starting and ending pixels location of APOGEE camera chips in the original 8575 pixels spectra
        |   - list[0] refers to the location where blue chips starts
        |   - list[1] refers to the location where blue chips ends
        |   - list[2] refers to the location where green chips starts
        |   - list[3] refers to the location where blue chips end
        |   - list[4] refers to the location where red chips starts
        |   - list[5] refers to the location where red chips ends
        |   - list[6] refers to the total number of pixels after deleting gap
    :rtype: list
    :History:
        | 2017-Nov-27 - Written - Henry Leung (University of Toronto)
        | 2017-Dec-16 - Updated - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)

    if dr == 11 or dr == 12:
        blue_start = 322
        blue_end = 3242
        green_start = 3648
        green_end = 6048
        red_start = 6412
        red_end = 8306
        total_pixel = 7214
    elif 13 <= dr <= 17:
        blue_start = 246
        blue_end = 3274
        green_start = 3585
        green_end = 6080
        red_start = 6344
        red_end = 8335
        total_pixel = 7514
    else:
        raise ValueError("Only DR11 to DR16 are supported")

    return [
        blue_start,
        blue_end,
        green_start,
        green_end,
        red_start,
        red_end,
        total_pixel,
    ]


def gap_delete(spectra, dr=None):
    """
    To delete the gap between APOGEE CCDs from the original 8575 pixels spectra

    :param spectra: The original 8575 pixels spectrum/spectra
    :type spectra: ndarray
    :param dr: data release
    :type dr: Union(int, NoneType)
    :return: Gap deleted spectrum/spectra
    :rtype: ndarray
    :History:
        | 2017-Oct-26 - Written - Henry Leung (University of Toronto)
        | 2017-Dec-16 - Updated - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)
    spectra = np.atleast_2d(spectra)
    info = chips_pix_info(dr=dr)

    if spectra.shape[1] != 8575 and spectra.shape[1] != info[6]:
        raise EnvironmentError("Are you sure you are giving astroNN APOGEE spectra?")
    if spectra.shape[1] != info[6]:
        spectra = spectra[
            :, np.r_[info[0] : info[1], info[2] : info[3], info[4] : info[5]]
        ]

    return spectra


def wavelength_solution(dr=None):
    """
    To return wavelegnth_solution, apStarWavegrid was provided by Jo Bovy's apogee tools (Toronto)

    :param dr: data release
    :type dr: Union(int, NoneType)
    :return:
        | lambda_blue, lambda_green, lambda_red which are 3 wavelength solution array
        |   - lambda_blue refers to the wavelength solution for each pixel in blue chips
        |   - lambda_green refers to the wavelength solution for each pixel in green chips
        |   - lambda_red refers to the wavelength solution for each pixel in red chips
    :rtype: ndarray
    :History:
        | 2017-Nov-20 - Written - Henry Leung (University of Toronto)
        | 2017-Dec-16 - Updated - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)
    info = chips_pix_info(dr=dr)

    apstar_wavegrid = 10.0 ** np.arange(
        4.179, 4.179 + 8575 * 6.0 * 10.0**-6.0, 6.0 * 10.0**-6.0
    )

    lambda_blue = apstar_wavegrid[info[0] : info[1]]
    lambda_green = apstar_wavegrid[info[2] : info[3]]
    lambda_red = apstar_wavegrid[info[4] : info[5]]

    return lambda_blue, lambda_green, lambda_red


def chips_split(spectra, dr=None):
    """
    To split APOGEE spectra into RGB chips, will delete the gap if detected

    :param spectra: APOGEE spectrum/spectra
    :type spectra: ndarray
    :param dr: data release
    :type dr: Union(int, NoneType)
    :return: 3 ndarrays which are spectra_blue, spectra_green, spectra_red
    :rtype: ndarray
    :History:
        | 2017-Nov-20 - Written - Henry Leung (University of Toronto)
        | 2017-Dec-17 - Updated - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)
    info = chips_pix_info(dr=dr)
    blue = info[1] - info[0]
    green = info[3] - info[2]
    red = info[5] - info[4]

    spectra = np.atleast_2d(spectra)

    if spectra.shape[1] == 8575:
        spectra = gap_delete(spectra, dr=dr)
        warnings.warn(
            "Raw spectra with gaps between detectors, gaps are removed automatically"
        )
    elif spectra.shape[1] == info[6]:
        pass
    else:
        raise EnvironmentError("Are you sure you are giving me APOGEE spectra?")

    spectra_blue = spectra[:, 0:blue]
    spectra_green = spectra[:, blue : (blue + green)]
    spectra_red = spectra[:, (blue + green) : (blue + green + red)]

    return spectra_blue, spectra_green, spectra_red


def bitmask_boolean(bitmask, target_bit):
    """
    Turn bitmask to boolean with provided bitmask array and target bit to mask

    :param bitmask: bitmask
    :type bitmask: ndarray
    :param target_bit: target bit to mask
    :type target_bit: list[int]
    :return: boolean array, True for clean, False for masked
    :rtype: ndarray[bool]
    :History: 2018-Feb-03 - Written - Henry Leung (University of Toronto)
    """
    target_bit = np.array(target_bit)
    target_bit = np.sum(2**target_bit)
    bitmask = np.atleast_2d(bitmask)
    boolean_output = np.zeros(bitmask.shape, dtype=bool)
    boolean_output[(bitmask & target_bit) != 0] = True
    return boolean_output


def bitmask_decompositor(bit):
    """
    To decompose a bit from bitmask array to individual bit

    :param bit: bitmask
    :type bit: int
    :return: boolean array, True for clean, False for masked
    :rtype: ndarray[bool]
    :History: 2018-Feb-03 - Written - Henry Leung (University of Toronto)
    """
    bitmask_num = int(bit)
    if bitmask_num < 0:
        raise ValueError(
            f"Your number ({bit}) is not valid, this value must not from a bitmask"
        )
    if bitmask_num == 0:
        print("0 corresponds to good pixel, thus this bit cannot be decomposed")
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


def continuum(spectra, spectra_err, cont_mask, deg=2):
    """
    Fit Chebyshev polynomials to the flux values in the continuum mask by chips.
    The resulting continuum will have the same shape as `fluxes`.

    :param spectra: spectra
    :type spectra: ndarray
    :param spectra_err: spectra uncertainty, same shape as spectra
    :type spectra_err: ndarray
    :param cont_mask: continuum mask
    :type cont_mask: ndarray[bool]
    :param deg: The degree of Chebyshev polynomial to use in each region, default is 2 which works the best so far
    :type deg: int
    :return: normalized spectra, normalized spectra uncertainty
    :rtype: ndarray, ndarray
    :History:
        | 2017-Dec-04 - Written - Henry Leung (University of Toronto)
        | 2017-Dec-16 - Update - Henry Leung (University of Toronto)
        | 2018-Mar-21 - Update - Henry Leung (University of Toronto)
    """
    spectra = np.atleast_2d(np.array(spectra))
    spectra_err = np.atleast_2d(np.array(spectra_err))
    flux_ivars = 1 / (
        np.square(np.array(spectra_err)) + 1e-8
    )  # for numerical stability

    pix_element = np.arange(spectra.shape[1])  # Array with size spectra

    for counter, (spectrum, spectrum_err, flux_ivar) in enumerate(
        zip(spectra, spectra_err, flux_ivars)
    ):
        no_nan_mask = ~np.isnan(spectrum[cont_mask])
        fit = np.polynomial.chebyshev.Chebyshev.fit(
            x=np.arange(spectrum.shape[0])[cont_mask][no_nan_mask],
            y=spectrum[cont_mask][no_nan_mask],
            w=flux_ivar[cont_mask][no_nan_mask],
            deg=deg,
        )
        spectra[counter] = spectrum / fit(pix_element)
        spectra_err[counter] = spectrum_err / fit(pix_element)

    return spectra, spectra_err


def apogee_continuum(
    spectra,
    spectra_err,
    cont_mask=None,
    deg=2,
    dr=None,
    bitmask=None,
    target_bit=None,
    mask_value=1.0,
):
    """
    It is designed only for apogee spectra by fitting Chebyshev polynomials to the flux values in the continuum mask
    by chips. The resulting continuum will have the same shape as `fluxes`.

    :param spectra: spectra
    :type spectra: ndarray
    :param spectra_err: spectra uncertainty, same shape as spectra
    :type spectra_err: ndarray
    :param cont_mask: continuum mask
    :type cont_mask: ndarray[bool]
    :param deg: The degree of Chebyshev polynomial to use in each region, default is 2 which works the best so far
    :type deg: int
    :param dr: apogee dr
    :type dr: int
    :param bitmask: bitmask array of the spectra, same shape as spectra
    :type bitmask: ndarray
    :param target_bit: a list of bit to be masked
    :type target_bit: Union(int, list[int], ndarray[int])
    :param mask_value: if a pixel is determined to be a bad pixel, this value will be used to replace that pixel flux
    :type mask_value: Union(int, float)
    :return: normalized spectra, normalized spectra uncertainty
    :rtype: ndarray, ndarray
    :History: 2018-Mar-21 - Written - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)

    spectra = gap_delete(spectra, dr=dr)
    flux_errs = gap_delete(spectra_err, dr=dr)

    spectra_blue, spectra_green, spectra_red = chips_split(spectra, dr=dr)
    yerrs_blue, yerrs_green, yerrs_red = chips_split(flux_errs, dr=dr)

    if cont_mask is None:
        maskpath = os.path.join(astroNN.data.datapath(), f"dr{dr}_contmask.npy")
        cont_mask = np.load(maskpath)

    con_mask_blue, con_mask_green, con_mask_red = chips_split(cont_mask, dr=dr)
    con_mask_blue, con_mask_green, con_mask_red = (
        con_mask_blue[0],
        con_mask_green[0],
        con_mask_red[0],
    )

    # Continuum chips by chips
    blue_spectra, blue_spectra_err = continuum(
        spectra_blue, yerrs_blue, cont_mask=con_mask_blue, deg=deg
    )
    green_spectra, green_spectra_err = continuum(
        spectra_green, yerrs_green, cont_mask=con_mask_green, deg=deg
    )
    red_spectra, red_spectra_err = continuum(
        spectra_red, yerrs_red, cont_mask=con_mask_red, deg=deg
    )

    normalized_spectra = np.concatenate(
        (blue_spectra, green_spectra, red_spectra), axis=1
    )
    normalized_spectra_err = np.concatenate(
        (blue_spectra_err, green_spectra_err, red_spectra_err), axis=1
    )

    # set negative flux and error as 0
    normalized_spectra[normalized_spectra < 0.0] = 0.0
    normalized_spectra_err[normalized_spectra < 0.0] = 0.0

    # set inf and nan as 0
    normalized_spectra[np.isinf(normalized_spectra)] = mask_value
    normalized_spectra[np.isnan(normalized_spectra)] = mask_value
    normalized_spectra_err[np.isinf(normalized_spectra)] = 0.0
    normalized_spectra_err[np.isnan(normalized_spectra)] = 0.0

    if bitmask is not None:
        bitmask = gap_delete(bitmask, dr=dr)
        if target_bit is None:
            target_bit = [0, 1, 2, 3, 4, 5, 6, 7, 12]

        mask = bitmask_boolean(bitmask, target_bit)
        normalized_spectra[mask] = mask_value
        normalized_spectra_err[mask] = mask_value

    return normalized_spectra, normalized_spectra_err


def aspcap_mask(elem, dr=None):
    """
    | To load ASPCAP elements window masks
    | DR14 Elements: ``'C', 'CI', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'K', 'Ca', 'TI', 'TiII', 'V', 'Cr', 'Mn',
                     'Fe', 'Co', 'Ni', 'Cu', 'Ge', 'Ce', 'Rb', 'Y', 'Nd'``

    :param elem: element name
    :type elem: str
    :param dr: apogee dr
    :type dr: int
    :return: mask
    :rtype: ndarray[bool]
    :History: 2018-Mar-24 - Written - Henry Leung (University of Toronto)
    """
    if elem.lower() == "c1":
        elem = "CI"

    elif elem.lower() == "ti2":
        elem = "TiII"

    dr = apogee_default_dr(dr=dr)

    if 14 <= dr <= 17:
        aspcap_code = "l31c"
        elem_list = [
            "C",
            "CI",
            "N",
            "O",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "K",
            "Ca",
            "TI",
            "TiII",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Ge",
            "Ce",
            "Rb",
            "Y",
            "Nd",
        ]
    else:
        raise ValueError("Only DR14-DR16 is supported currently")

    masks = np.load(
        os.path.join(astroNN.data.datapath(), f"aspcap_{aspcap_code}_masks.npy")
    )

    try:
        # turn everything to lowercase to avoid case-related issue
        index = [x.lower() for x in elem_list].index(elem.lower())
    except ValueError:
        # nicely handle if element not found
        print(
            f"Element not found, the only elements for dr{dr} supported are {elem_list}"
        )
        return None

    return [(masks & 2**index) != 0][0]
