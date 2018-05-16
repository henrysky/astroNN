# ---------------------------------------------------------#
#   astroNN.gaia.gaia_shared: shared functions for apogee
# ---------------------------------------------------------#

import os
import warnings

import numpy as np
from astropy import units as u
from astropy import constants
from astroNN.config import MAGIC_NUMBER

default_parallax_unit = u.mas
solar_absmag = -2.5 * np.log10(constants.L_sun.value / constants.L_bol0.value)  # Sun's absmag


def gaia_env():
    """
    Get Gaia environment variable

    :return: Path to Gaia Data
    :rtype: str
    :History: 2017-Oct-26 - Written - Henry Leung (University of Toronto)
    """
    from astroNN.config import ENVVAR_WARN_FLAG
    _GAIA = os.getenv('GAIA_TOOLS_DATA')
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
        dr = 1
        print(f'dr is not provided, using default dr={dr}')
    else:
        pass
    return dr


def mag_to_fakemag(mag, parallax, parallax_err=None):
    """
    To convert apparent magnitude to astroNN fakemag, Magic Number will be preserved

    :param mag: apparent magnitude
    :type mag: Union[float, ndarray]
    :param parallax: parallax (mas) or with astropy so astroNN will convert to appropriate units
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
        parallax = parallax.to(default_parallax_unit)
        if parallax_err is not None:
            if not isinstance(parallax_err, u.Quantity):
                # assume parallax error carry the same original unit as parallax if no units detected
                parallax_err = (parallax_err * original_parallax_unit).to(default_parallax_unit).value
            if isinstance(parallax_err, u.Quantity):
                parallax_err = parallax_err.to(default_parallax_unit).value
        print(
            f'Please be advised that astroNN fakemag function expects {default_parallax_unit.name}, astroNN has '
            f'corrected the unit according to astropy unit framework')
    else:
        print(f'Please be advised that astroNN fakemag is parallax({default_parallax_unit.name}) * 10 ** (0.2 * mag)')

    mag = np.array(mag)
    parallax_unitless = np.array(parallax)  # Take the value as we cant apply pow() to astropy unit

    magic_idx = ((parallax_unitless == MAGIC_NUMBER) | (mag == MAGIC_NUMBER) | (mag < -90.))  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        fakemag = parallax_unitless * (10. ** (0.2 * mag))
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
    :param parallax: parallax (mas) or with astropy so astroNN will convert to appropriate units
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
        parallax = parallax.to(default_parallax_unit)
        if parallax_err is not None:
            if not isinstance(parallax_err, u.Quantity):
                # assume parallax error carry the same original unit as parallax if no units detected
                parallax_err = (parallax_err * original_parallax_unit).to(default_parallax_unit).value
            if isinstance(parallax_err, u.Quantity):
                parallax_err = parallax_err.to(default_parallax_unit).value
        print(f'Please be advised that astroNN mag_to_absmag() expects {default_parallax_unit.name}, '
              f'astroNN has corrected the unit according to astropy unit framework')
    else:
        print(f'Please be advised that astroNN mag_to_absmag expects parallax in {default_parallax_unit.name}')

    mag = np.array(mag)
    parallax_unitless = np.array(parallax)  # Take the value as we cant apply log10 to astropy unit

    magic_idx = ((parallax_unitless == MAGIC_NUMBER) | (mag == MAGIC_NUMBER) | (mag < -90.))  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        absmag = mag + 5. * (np.log10(parallax_unitless) - 2.)

    if parallax_unitless.shape != ():  # check if its only 1 element
        absmag[magic_idx] = MAGIC_NUMBER
    else:
        absmag = MAGIC_NUMBER if magic_idx == [1] else absmag
    if parallax_err is None:
        return absmag
    else:
        absmag_err = 5. * np.abs(parallax_err / (parallax_unitless * absmag))
        if parallax_unitless.shape != ():  # check if its only 1 element
            absmag_err[magic_idx] = MAGIC_NUMBER
        else:
            absmag_err = MAGIC_NUMBER if magic_idx == [1] else absmag_err
        return absmag, absmag_err


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
    magic_idx = ((absmag == MAGIC_NUMBER) | (mag == MAGIC_NUMBER))  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        pc = (10. ** (((mag - absmag) / 5.) + 1.))

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
    # treat negative fakemag as MAGIC_NUMBER
    magic_idx = [(fakemag == MAGIC_NUMBER) | (fakemag < 0.)]  # check for magic number and negative fakemag

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        absmag = 5. * (np.log10(fakemag) - 2.)

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
    magic_idx = (absmag == MAGIC_NUMBER)  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        fakemag = 10. ** (0.2 * absmag + 2.)
    if fakemag.shape != ():  # check if its only 1 element
        fakemag[magic_idx] = MAGIC_NUMBER
    else:  # for float
        fakemag = MAGIC_NUMBER if magic_idx == [1] else fakemag
    return fakemag


def fakemag_to_pc(fakemag, mag, fakemag_err=None):
    """
    To convert fakemag to parsec, Magic Number will be preserved

    :param fakemag: astroNN fakemag
    :type fakemag: Union[float, ndarray]
    :param mag: apparent magnitude
    :type mag: Union[float, ndarray]
    :param fakemag_err: Optional, fakemag_err
    :type fakemag_err: Union[NoneType, float, ndarray]
    :return: array of parsec with astropy Quantity (with additional return of propagated error if fakemag_err is provided)
    :rtype: astropy Quantity
    :History: 2018-Jan-31 - Written - Henry Leung (University of Toronto)
    """
    fakemag = np.array(fakemag)
    mag = np.array(mag)
    # treat negative fakemag as MAGIC_NUMBER, check for magic number and negative fakemag
    magic_idx = ((fakemag == MAGIC_NUMBER) | (mag == MAGIC_NUMBER) | (fakemag < 0.))

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        pc = 1000. * (10. ** (0.2 * mag)) / fakemag
    if fakemag.shape != ():  # check if its only 1 element
        pc[magic_idx] = MAGIC_NUMBER
    else:  # for float
        pc = MAGIC_NUMBER if magic_idx == [1] else pc

    if fakemag_err is None:
        return pc * u.parsec
    else:
        pc_err = (fakemag_err / fakemag) * pc
        if fakemag.shape != ():  # check if its only 1 element
            pc_err[magic_idx] = MAGIC_NUMBER
        else:  # for float
            pc_err = MAGIC_NUMBER if magic_idx == [1] else pc_err
        return pc * u.parsec, pc_err * u.parsec


def fakemag_to_logsol(fakemag):
    """
    | To convert fakemag to log solar luminosity, negative fakemag will be converted to MAGIC_NUMBER because of fakemag
    | cannnot be negative in physical world

    :param fakemag: astroNN fakemag
    :type fakemag: Union[float, ndarray]
    :return: log solar luminosity
    :rtype: Union[float, ndarray]
    :History: 2018-May-06 - Written - Henry Leung (University of Toronto)
    """
    fakemag = np.array(fakemag)
    magic_idx = [(fakemag == MAGIC_NUMBER) | (fakemag < 0.)]  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        logsol_lum = 0.4 * (solar_absmag - fakemag_to_absmag(fakemag))

    if logsol_lum.shape != ():  # check if its only 1 element
        logsol_lum[magic_idx] = MAGIC_NUMBER
    else:  # for float
        logsol_lum = MAGIC_NUMBER if magic_idx == [1] else logsol_lum
    return logsol_lum


def absmag_to_logsol(absmag):
    """
    To convert absmag to log solar luminosity

    :param absmag: absolute magnitude
    :type absmag: Union[float, ndarray]
    :return: log solar luminosity
    :rtype: Union[float, ndarray]
    :History: 2018-May-06 - Written - Henry Leung (University of Toronto)
    """
    absmag = np.array(absmag)
    magic_idx = (absmag == MAGIC_NUMBER)  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        logsol_lum = 0.4 * (solar_absmag - absmag)

    if logsol_lum.shape != ():  # check if its only 1 element
        logsol_lum[magic_idx] = MAGIC_NUMBER
    else:  # for float
        logsol_lum = MAGIC_NUMBER if magic_idx == [1] else logsol_lum
    return logsol_lum


def logsol_to_fakemag(logsol):
    """
    | To convert log solar luminosity to fakemag, negative fakemag will be converted to MAGIC_NUMBER because of fakemag
    | cannnot be negative in physical world

    :param logsol: log solar luminosity
    :type logsol: Union[float, ndarray]
    :return: astroNN fakemag
    :rtype: Union[float, ndarray]
    :History: 2018-May-06 - Written - Henry Leung (University of Toronto)
    """
    logsol = np.array(logsol)
    magic_idx = [(logsol == MAGIC_NUMBER)]  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        fakemag = absmag_to_fakemag(solar_absmag - logsol / 0.4)

    if fakemag.shape != ():  # check if its only 1 element
        fakemag[magic_idx] = MAGIC_NUMBER
    else:  # for float
        fakemag = MAGIC_NUMBER if magic_idx == [1] else fakemag
    return fakemag


def logsol_to_absmag(logsol):
    """
    | To convert log solar luminosity to absmag, negative fakemag will be converted to MAGIC_NUMBER because of fakemag
    | cannnot be negative in physical world

    :param logsol: log solar luminosity
    :type logsol: Union[float, ndarray]
    :return: absmag
    :rtype: Union[float, ndarray]
    :History: 2018-May-06 - Written - Henry Leung (University of Toronto)
    """
    logsol = np.array(logsol)
    magic_idx = [(logsol == MAGIC_NUMBER)]  # check for magic number

    with warnings.catch_warnings():  # suppress numpy Runtime warning caused by MAGIC_NUMBER
        warnings.simplefilter("ignore")
        absmag = solar_absmag - logsol / 0.4

    if absmag.shape != ():  # check if its only 1 element
        absmag[magic_idx] = MAGIC_NUMBER
    else:  # for float
        absmag = MAGIC_NUMBER if magic_idx == [1] else absmag
    return absmag


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
    extinction[extinction < -10.] = 0.  # extinction cannot be that negative, if yes then assume no extinction
    magic_idx = ((mag == MAGIC_NUMBER) | (mag < -90.))  # check for magic number

    mag_ec = mag - extinction
    if mag_ec.shape != ():  # check if its only 1 element
        mag_ec[magic_idx] = MAGIC_NUMBER
        return mag_ec
    else:
        return MAGIC_NUMBER if magic_idx == [1] else mag_ec
