# ---------------------------------------------------------#
#   astroNN.gaia.gaia_shared: shared functions for apogee
# ---------------------------------------------------------#

import os
import warnings

import numpy as np
from astropy import units as u

from astroNN.config import MAGIC_NUMBER

default_parallax_unit = u.mas

# Sun's absmag in different Johnson/Cousins/2MASS/Sloan bands in Vega system
# sun abs mag in pass bands source: https://arxiv.org/pdf/1804.07788.pdf --> Table 3
solar_absmag_bands = {
    "U": 5.61,  # Johnson U
    "B": 5.44,  # Johnson B
    "V": 4.81,  # Johnson V
    "R": 4.43,  # Cousins R
    "I": 4.10,  # Cousins I
    "J": 3.67,  # 2MASS J
    "H": 3.32,  # 2MASS H
    "K": 3.27,  # 2MASS Ks
    "Ks": 3.27,  # 2MASS Ks
    "u": 5.49,  # SDSS u
    "g": 5.23,  # SDSS g
    "r": 4.53,  # SDSS r
    "i": 4.19,  # SDSS i
    "z": 4.01,  # SDSS z
    "G": 4.67,  # Gaia G, https://arxiv.org/pdf/1806.01953.pdf
}  


def gaia_env():
    """
    Get Gaia environment variable

    :return: Path to Gaia Data
    :rtype: str
    :History: 2017-Oct-26 - Written - Henry Leung (University of Toronto)
    """
    from astroNN.config import ENVVAR_WARN_FLAG

    _GAIA = os.getenv("GAIA_TOOLS_DATA")
    if _GAIA is None and ENVVAR_WARN_FLAG is True:
        print("WARNING! Gaia environment variable GAIA_TOOLS_DATA not set")
    return _GAIA


def gaia_default_dr(dr=None):
    """
    Check if dr argument is provided, if none then use default

    :param dr: Gaia DR
    :type dr: Union[NoneType, int]
    :return: Gaia DR
    :rtype: int
    :History: 2017-Oct-26 - Written - Henry Leung (University of Toronto)
    """
    if dr is None:
        dr = 2
        print(f"dr is not provided, using default dr={dr}")
    else:
        pass
    return dr


def mag_to_fakemag(mag, parallax, parallax_err=None):
    """
    To convert apparent magnitude to astroNN fakemag, Magic Number will be preserved

    :param mag: apparent magnitude
    :type mag: Union[float, ndarray]
    :param parallax: parallax (mas) or with astropy(can be distance with units) so astroNN will convert to appropriate units
    :type parallax: Union[float, ndarray, astropy Quantity]
    :param parallax_err: parallax_error (mas) or with astropy so astroNN will convert to appropriate units
    :type parallax_err: Union[NoneType, float, ndarray, astropy Quantity]
    :return: astroNN fakemag, with addition (with additional return of propagated error if parallax_err is provided)
    :rtype: Union[float, ndarray]
    :History: 2017-Oct-14 - Written - Henry Leung (University of Toronto)
    """
    # Check unit if available
    if isinstance(parallax, u.Quantity):
        original_parallax_unit = parallax.unit
        parallax = parallax.to(default_parallax_unit, equivalencies=u.parallax())
        if parallax_err is not None:
            if not isinstance(parallax_err, u.Quantity):
                # assume parallax error carry the same original unit as parallax if no units detected
                parallax_err = (
                    (parallax_err * original_parallax_unit)
                    .to(default_parallax_unit)
                    .value
                )
            if isinstance(parallax_err, u.Quantity):
                parallax_err = parallax_err.to(default_parallax_unit).value

    mag = np.array(mag)
    parallax_unitless = np.array(
        parallax
    )  # Take the value as we cant apply pow() to astropy unit

    magic_idx = (
        (parallax_unitless == MAGIC_NUMBER)
        | (mag == MAGIC_NUMBER)
        | (mag < -90.0)
        | np.isnan(parallax_unitless)
        | np.isnan(mag)
    )  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        fakemag = parallax_unitless * (10.0 ** (0.2 * mag))
    if parallax_unitless.shape != ():  # check if its only 1 element
        fakemag[magic_idx] = MAGIC_NUMBER
    else:
        fakemag = MAGIC_NUMBER if magic_idx == [1] else fakemag

    if parallax_err is None:
        return fakemag
    else:
        fakemag_err = np.abs((parallax_err / parallax_unitless) * fakemag)
        if parallax_unitless.shape != ():  # check if its only 1 element
            fakemag_err[magic_idx] = MAGIC_NUMBER
        else:
            fakemag_err = MAGIC_NUMBER if magic_idx == [1] else fakemag_err
        return fakemag, fakemag_err


def mag_to_absmag(mag, parallax, parallax_err=None):
    """
    To convert apparent magnitude to absolute magnitude, Magic Number will be preserved

    :param mag: apparent magnitude
    :type mag: Union[float, ndarray]
    :param parallax: parallax (mas) or with astropy (can be distance with units) so astroNN will convert to appropriate units
    :type parallax: Union[float, ndarray, astropy Quantity]
    :param parallax_err: parallax_error (mas) or with astropy so astroNN will convert to appropriate units
    :type parallax_err: Union[NoneType, float, ndarray, astropy Quantity]
    :return: absolute magnitude  (with additional return of propagated error if parallax_err is provided)
    :rtype: Union[float, ndarray]
    :History: 2017-Oct-14 - Written - Henry Leung (University of Toronto)
    """
    # Check unit if available
    if isinstance(parallax, u.Quantity):
        original_parallax_unit = parallax.unit
        parallax = parallax.to(default_parallax_unit, equivalencies=u.parallax())
        if parallax_err is not None:
            if not isinstance(parallax_err, u.Quantity):
                # assume parallax error carry the same original unit as parallax if no units detected
                parallax_err = (
                    (parallax_err * original_parallax_unit)
                    .to(default_parallax_unit)
                    .value
                )
            if isinstance(parallax_err, u.Quantity):
                parallax_err = parallax_err.to(default_parallax_unit).value
    #     warnings.warn(f'Please be advised that astroNN mag_to_absmag() expects {default_parallax_unit.name}, '
    #           f'astroNN has corrected the unit according to astropy unit framework')
    # else:
    #     warnings.warn(f'Please be advised that astroNN mag_to_absmag expects parallax in {default_parallax_unit.name}')

    mag = np.array(mag)
    parallax_unitless = np.array(
        parallax
    )  # Take the value as we cant apply log10 to astropy unit

    magic_idx = (
        (parallax_unitless == MAGIC_NUMBER)
        | (mag == MAGIC_NUMBER)
        | (mag < -90.0)
        | np.isnan(parallax_unitless)
        | np.isnan(mag)
    )  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        absmag = mag + 5.0 * (np.log10(parallax_unitless) - 2.0)

    if parallax_unitless.shape != ():  # check if its only 1 element
        absmag[magic_idx] = MAGIC_NUMBER
    else:
        absmag = MAGIC_NUMBER if magic_idx == [1] else absmag
    if parallax_err is None:
        return absmag
    else:
        absmag_err = 5.0 * np.abs(parallax_err / (parallax_unitless * absmag))
        if parallax_unitless.shape != ():  # check if its only 1 element
            absmag_err[magic_idx] = MAGIC_NUMBER
        else:
            absmag_err = MAGIC_NUMBER if magic_idx == [1] else absmag_err
        return absmag, absmag_err


# noinspection PyUnresolvedReferences
def absmag_to_pc(absmag, mag):
    """
    To convert absolute magnitude to parsec, Magic Number will be preserved

    :param absmag: absolute magnitude
    :type absmag: Union[float, ndarray]
    :param mag: apparent magnitude
    :type mag: Union[float, ndarray]
    :return: parsec
    :rtype: astropy Quantity
    :History: 2017-Nov-16 - Written - Henry Leung (University of Toronto)
    """
    absmag = np.array(absmag)
    mag = np.array(mag)
    magic_idx = (
        (absmag == MAGIC_NUMBER)
        | (mag == MAGIC_NUMBER)
        | np.isnan(absmag)
        | np.isnan(mag)
    )  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        pc = 10.0 ** (((mag - absmag) / 5.0) + 1.0)

    if absmag.shape != ():  # check if its only 1 element
        pc[magic_idx] = MAGIC_NUMBER
        return pc * u.parsec
    else:
        return (MAGIC_NUMBER if magic_idx == [1] else pc) * u.parsec


def fakemag_to_absmag(fakemag):
    """
    To convert fakemag to absmag, Magic Number will be preserved

    :param fakemag: eastroNN fakemag
    :type fakemag: Union[float, ndarray]
    :return: absolute magnitude
    :rtype: Union[float, ndarray]
    :History: 2018-Jan-31 - Written - Henry Leung (University of Toronto)
    """
    fakemag = np.array(fakemag)
    # treat non-negative fakemag as MAGIC_NUMBER
    magic_idx = (
        (fakemag == MAGIC_NUMBER) | (fakemag <= 0.0) | np.isnan(fakemag)
    )  # check for magic number and negative fakemag

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        absmag = 5.0 * (np.log10(fakemag) - 2.0)

    if absmag.shape != ():  # check if its only 1 element
        absmag[magic_idx] = MAGIC_NUMBER
    else:  # for float
        absmag = MAGIC_NUMBER if magic_idx == [1] else absmag
    return absmag


def absmag_to_fakemag(absmag):
    """
    To convert absmag to fakemag, Magic Number will be preserved

    :param absmag: absolute magnitude
    :type absmag: Union[float, ndarray]
    :return: astroNN fakemag
    :rtype: Union[float, ndarray]
    :History: 2018-Jan-31 - Written - Henry Leung (University of Toronto)
    """
    absmag = np.array(absmag)
    magic_idx = (absmag == MAGIC_NUMBER) | np.isnan(absmag)  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        fakemag = 10.0 ** (0.2 * absmag + 2.0)
    if fakemag.shape != ():  # check if its only 1 element
        fakemag[magic_idx] = MAGIC_NUMBER
    else:  # for float
        fakemag = MAGIC_NUMBER if magic_idx == [1] else fakemag
    return fakemag


# noinspection PyUnresolvedReferences
def fakemag_to_pc(fakemag, mag, fakemag_err=None):
    """
    To convert fakemag to parsec, Magic Number will be preserved

    :param fakemag: astroNN fakemag
    :type fakemag: Union[float, ndarray]
    :param mag: apparent magnitude
    :type mag: Union[float, ndarray]
    :param fakemag_err: Optional, fakemag_err
    :type fakemag_err: Union[NoneType, float, ndarray]
    :return: array of pc with astropy Quantity (with additional return of propagated error if fakemag_err is provided)
    :rtype: astropy Quantity
    :History: 2018-Jan-31 - Written - Henry Leung (University of Toronto)
    """
    fakemag = np.array(fakemag)
    mag = np.array(mag)
    # treat non-positive fakemag as MAGIC_NUMBER, check for magic number and negative fakemag
    magic_idx = (
        (fakemag == MAGIC_NUMBER)
        | (mag == MAGIC_NUMBER)
        | (fakemag <= 0.0)
        | np.isnan(fakemag)
        | np.isnan(mag)
    )

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        pc = 1000.0 * (10.0 ** (0.2 * mag)) / fakemag
    if fakemag.shape != ():  # check if its only 1 element
        pc[magic_idx] = MAGIC_NUMBER
    else:  # for float
        pc = MAGIC_NUMBER if magic_idx == [1] else pc

    if fakemag_err is None:
        return pc * u.parsec
    else:
        with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
            warnings.simplefilter("ignore")
            pc_err = (fakemag_err / fakemag) * pc
        if fakemag.shape != ():  # check if its only 1 element
            pc_err[magic_idx] = MAGIC_NUMBER
        else:  # for float
            pc_err = MAGIC_NUMBER if magic_idx == [1] else pc_err
        return pc * u.parsec, pc_err * u.parsec


# noinspection PyUnresolvedReferences
def fakemag_to_parallax(fakemag, mag, fakemag_err=None):
    """
    To convert fakemag to parallax, Magic Number will be preserved

    :param fakemag: astroNN fakemag
    :type fakemag: Union[float, ndarray]
    :param mag: apparent magnitude
    :type mag: Union[float, ndarray]
    :param fakemag_err: Optional, fakemag_err
    :type fakemag_err: Union[NoneType, float, ndarray]
    :return: array of parallax in mas with astropy Quantity (with additional return of propagated error if fakemag_err is provided)
    :rtype: astropy Quantity
    :History: 2018-Aug-11 - Written - Henry Leung (University of Toronto)
    """
    fakemag = np.array(fakemag)
    mag = np.array(mag)
    # treat non-positive fakemag as MAGIC_NUMBER, check for magic number and negative fakemag
    magic_idx = (
        (fakemag == MAGIC_NUMBER)
        | (mag == MAGIC_NUMBER)
        | (fakemag <= 0.0)
        | np.isnan(fakemag)
        | np.isnan(mag)
    )

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        parallax = fakemag / (10.0 ** (0.2 * mag))
    if fakemag.shape != ():  # check if its only 1 element
        parallax[magic_idx] = MAGIC_NUMBER
    else:  # for float
        parallax = MAGIC_NUMBER if magic_idx == [1] else parallax

    if fakemag_err is None:
        return parallax * u.mas
    else:
        with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
            warnings.simplefilter("ignore")
            parallax_err = (fakemag_err / fakemag) * parallax
        if fakemag.shape != ():  # check if its only 1 element
            parallax_err[magic_idx] = MAGIC_NUMBER
        else:  # for float
            parallax_err = MAGIC_NUMBER if magic_idx == [1] else parallax_err
        return parallax * u.mas, parallax_err * u.mas


def fakemag_to_logsol(fakemag, band="K"):
    """
    | To convert fakemag to log10 solar luminosity, negative fakemag will be converted to MAGIC_NUMBER because of
    | fakemag cannot be negative in physical world

    :param fakemag: astroNN fakemag
    :type fakemag: Union[float, ndarray]
    :param band: band of your fakemag to use with
    :type band: str(['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K','u', 'g', 'r', 'i', 'z'])
    :return: log solar luminosity
    :rtype: Union[float, ndarray]
    :History: 2018-May-06 - Written - Henry Leung (University of Toronto)
    """
    fakemag = np.array(fakemag)
    magic_idx = (
        (fakemag == MAGIC_NUMBER) | (fakemag <= 0.0) | np.isnan(fakemag)
    )  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        log10sol_lum = np.array(
            0.4 * (solar_absmag_bands[band] - fakemag_to_absmag(fakemag))
        )

    if log10sol_lum.shape != ():  # check if its only 1 element
        log10sol_lum[magic_idx] = MAGIC_NUMBER
    else:  # for float
        log10sol_lum = MAGIC_NUMBER if magic_idx == [1] else log10sol_lum
    return log10sol_lum


def absmag_to_logsol(absmag, band="K"):
    """
    To convert absmag to log10 solar luminosity

    :param absmag: absolute magnitude
    :type absmag: Union[float, ndarray]
    :param band: band of your absmag to use with
    :type band: str(['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K','u', 'g', 'r', 'i', 'z'])
    :return: log solar luminosity
    :rtype: Union[float, ndarray]
    :History: 2018-May-06 - Written - Henry Leung (University of Toronto)
    """
    absmag = np.array(absmag)
    magic_idx = (absmag == MAGIC_NUMBER) | np.isnan(absmag)  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        log10sol_lum = np.array(0.4 * (solar_absmag_bands[band] - absmag))

    if log10sol_lum.shape != ():  # check if its only 1 element
        log10sol_lum[magic_idx] = MAGIC_NUMBER
    else:  # for float
        log10sol_lum = MAGIC_NUMBER if magic_idx == [1] else log10sol_lum
    return log10sol_lum


def logsol_to_fakemag(logsol, band="K"):
    """
    | To convert log10 solar luminosity to fakemag, negative fakemag will be converted to MAGIC_NUMBER because of fakemag
    | cannot be negative in physical world

    :param logsol: log solar luminosity
    :type logsol: Union[float, ndarray]
    :param band: band of your fakemag to use with
    :type band: str(['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K','u', 'g', 'r', 'i', 'z'])
    :return: astroNN fakemag
    :rtype: Union[float, ndarray]
    :History: 2018-May-06 - Written - Henry Leung (University of Toronto)
    """
    logsol = np.array(logsol)
    magic_idx = (logsol == MAGIC_NUMBER) | np.isnan(logsol)  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        fakemag = absmag_to_fakemag(solar_absmag_bands[band] - logsol / 0.4)

    if fakemag.shape != ():  # check if its only 1 element
        fakemag[magic_idx] = MAGIC_NUMBER
    else:  # for float
        fakemag = MAGIC_NUMBER if magic_idx == [1] else fakemag
    return fakemag


def logsol_to_absmag(logsol, band="K"):
    """
    | To convert log10 solar luminosity to absmag, negative fakemag will be converted to MAGIC_NUMBER because of fakemag
    | cannot be negative in physical world

    :param logsol: log solar luminosity
    :type logsol: Union[float, ndarray]
    :param band: band of your absmag to use with
    :type band: str(['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K','u', 'g', 'r', 'i', 'z'])
    :return: absmag
    :rtype: Union[float, ndarray]
    :History: 2018-May-06 - Written - Henry Leung (University of Toronto)
    """
    logsol = np.array(logsol)
    magic_idx = (logsol == MAGIC_NUMBER) | np.isnan(logsol)  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        absmag = solar_absmag_bands[band] - logsol / 0.4

    if absmag.shape != ():  # check if its only 1 element
        absmag[magic_idx] = MAGIC_NUMBER
    else:  # for float
        absmag = MAGIC_NUMBER if magic_idx == [1] else absmag
    return absmag


# noinspection PyUnresolvedReferences
def fakemag_to_mag(fakemag, pc, pc_err=None):
    """
    To convert apparent magnitude to astroNN fakemag, Magic Number will be preserved

    :param fakemag: fakemag
    :type fakemag: Union[float, ndarray]
    :param pc: parsec or with astropy (can be parallax with units) so astroNN will convert to appropriate units
    :type pc: Union[float, ndarray, astropy Quantity]
    :param pc_error: parsec uncertainty or with astropy so astroNN will convert to appropriate units
    :type pc_error: Union[NoneType, float, ndarray, astropy Quantity]
    :return: astroNN fakemag, with addition (with additional return of propagated error if parallax_err is provided)
    :rtype: Union[float, ndarray]
    :History: 2018-Aug-1 - Written - Henry Leung (University of Toronto)
    """
    # Check unit if available
    if isinstance(pc, u.Quantity):
        original_parallax_unit = pc.unit
        pc = pc.to(u.parsec, equivalencies=u.parallax())
        if pc_err is not None:
            if not isinstance(pc_err, u.Quantity):
                # assume parallax error carry the same original unit as parallax if no units detected
                pc_err = (pc_err * original_parallax_unit).to(u.parsec).value
            if isinstance(pc_err, u.Quantity):
                pc_err = pc_err.to(u.parsec).value

    fakemag = np.array(fakemag)
    pc_unitless = np.array(pc)  # Take the value as we cant apply pow() to astropy unit

    magic_idx = (
        (pc_unitless == MAGIC_NUMBER)
        | (fakemag == MAGIC_NUMBER)
        | (fakemag <= 0.0)
        | np.isnan(pc_unitless)
        | np.isnan(fakemag)
    )  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        mag = np.log10((pc_unitless / 1000) * fakemag) / 0.2
    if pc_unitless.shape != ():  # check if its only 1 element
        mag[magic_idx] = MAGIC_NUMBER
    else:
        fakemag = MAGIC_NUMBER if magic_idx == [1] else fakemag

    if pc_err is None:
        return mag
    else:
        mag_err = np.abs((pc_err / pc) * fakemag)
        if pc_unitless.shape != ():  # check if its only 1 element
            mag_err[magic_idx] = MAGIC_NUMBER
        else:
            mag_err = MAGIC_NUMBER if magic_idx == [1] else mag_err
        return mag, mag_err


def extinction_correction(mag, extinction):
    """
    To correct magnitude with extinction, this function assumes extinction is at the same wavelength as the magnitude
    you have provided

    :param mag: apparent magnitude
    :type mag: Union[float, ndarray]
    :param extinction: extinction
    :type extinction: Union[float, ndarray]
    :return: corrected magnitude
    :rtype: Union[float, ndarray]
    :History: 2018-May-13 - Written - Henry Leung (University of Toronto)
    """
    mag = np.array(mag)
    extinction = np.array(extinction)
    extinction[
        extinction < -1.0
    ] = 0.0  # extinction cannot be that negative, if yes then assume no extinction
    magic_idx = (
        (mag == MAGIC_NUMBER) | (mag < -90.0) | np.isnan(mag)
    )  # check for magic number

    mag_ec = mag - extinction
    if mag_ec.shape != ():  # check if its only 1 element
        mag_ec[magic_idx] = MAGIC_NUMBER
        return mag_ec
    else:
        return MAGIC_NUMBER if magic_idx == [1] else mag_ec
