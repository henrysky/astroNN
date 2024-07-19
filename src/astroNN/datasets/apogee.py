import numpy as np
from astropy import units as u
from astropy.io import fits

from astroNN.apogee import allstar
from astroNN.apogee.downloader import apogee_distances, apogee_rc
from astroNN.gaia import mag_to_absmag, mag_to_fakemag, extinction_correction
from astroquery.vizier import Vizier


def load_apogee_distances(
    dr=None, unit="distance", cuts=True, extinction=True, keepdims=False
):
    """
    Load apogee distances (absolute magnitude from stellar model)

    :param dr: Apogee DR
    :type dr: int
    :param unit: which unit you want to get back

                   - "absmag" for absolute magnitude
                   - "fakemag" for fake magnitude
                   - "distance" for distance in parsec
    :type unit: string
    :param cuts: Whether to cut bad data (negative parallax and percentage error more than 20%), or a float to set the threshold
    :type cuts: Union[boolean, float]
    :param extinction: Whether to take extinction into account, only affect when unit is NOT 'distance'
    :type extinction: bool
    :param keepdims: Whether to preserve indices the same as APOGEE allstar DR14, no effect when cuts=False, set to -9999 for bad indices when cuts=True keepdims=True
    :type keepdims: boolean
    :return: numpy array of ra, dec, array, err_array
    :rtype: ndarrays
    :History:
        | 2018-Jan-25 - Written - Henry Leung (University of Toronto)
        | 2021-Jan-29 - Updated - Henry Leung (University of Toronto)
    """
    fullfilename = apogee_distances(dr=dr)

    with fits.open(fullfilename) as F:
        hdulist = F[1].data
        # Convert kpc to pc
        distance = hdulist["BPG_dist50"] * 1000
        dist_err = (hdulist["BPG_dist84"] - hdulist["BPG_dist16"]) * 1000

    allstarfullpath = allstar(dr=dr)

    with fits.open(allstarfullpath) as F:
        k_mag = F[1].data["K"]
        if extinction:
            k_mag = extinction_correction(k_mag, F[1].data["AK_TARG"])
        ra = F[1].data["RA"]
        dec = F[1].data["DEC"]

    # Bad index refers to nan index
    bad_index = np.argwhere(np.isnan(distance))

    if unit == "distance":
        # removed astropy units because of -9999. is dimensionless, will have issues
        output = distance
        output_err = dist_err

    elif unit == "absmag":
        absmag, absmag_err = mag_to_absmag(
            k_mag, 1 / distance * u.arcsec, (1 / distance) * (dist_err / distance)
        )
        output = absmag
        output_err = absmag_err

    elif unit == "fakemag":
        # fakemag requires parallax (mas)
        fakemag, fakemag_err = mag_to_fakemag(
            k_mag, 1000 / distance * u.mas, (1000 / distance) * (dist_err / distance)
        )
        output = fakemag
        output_err = fakemag_err

    else:
        raise ValueError("Unknown unit")

    # Set the nan index to -9999. as they are bad and unknown. Not magic_number as this is an APOGEE dataset
    output[bad_index], output_err[bad_index] = -9999.0, -9999.0
    if cuts is False:
        pass
    else:
        distance[bad_index], dist_err[bad_index] = -9999.0, -9999.0
        good_idx = (dist_err / distance < (0.2 if cuts is True else cuts)) & (
            distance != -9999.0
        )

        if not keepdims:
            ra = ra[good_idx]
            dec = dec[good_idx]
            output = output[good_idx]
            output_err = output_err[good_idx]
        else:
            output[(dist_err / distance > (0.2 if cuts is True else cuts))] = -9999.0
            output_err[
                (dist_err / distance > (0.2 if cuts is True else cuts))
            ] = -9999.0

    return ra, dec, output, output_err


def load_apogee_rc(dr=None, unit="distance", extinction=True):
    """
    Load apogee red clumps (absolute magnitude measurement)

    :param dr: Apogee DR
    :type dr: int
    :param unit: which unit you want to get back

                   - "absmag" for k-band absolute magnitude
                   - "fakemag" for k-band fake magnitude
                   - "distance" for distance in parsec
    :type unit: string
    :param extinction: Whether to take extinction into account, only affect when unit is NOT 'distance'
    :type extinction: bool
    :return: numpy array of ra, dec, array
    :rtype: ndarrays
    :History:
        | 2018-Jan-21 - Written - Henry Leung (University of Toronto)
        | 2018-May-12 - Updated - Henry Leung (University of Toronto)
    """
    fullfilename = apogee_rc(dr=dr)

    with fits.open(fullfilename) as F:
        hdulist = F[1].data
        ra = hdulist["RA"]
        dec = hdulist["DEC"]
        rc_dist = hdulist["RC_DIST"]
        rc_parallax = (1 / rc_dist) * u.mas  # Convert kpc to parallax in mas
        k_mag = hdulist["K"]
        if extinction:
            k_mag = extinction_correction(k_mag, hdulist["AK_TARG"])

    if unit == "distance":
        output = rc_dist * 1000

    elif unit == "absmag":
        absmag = mag_to_absmag(k_mag, rc_parallax)
        output = absmag

    elif unit == "fakemag":
        # fakemag requires parallax (mas)
        fakemag = mag_to_fakemag(k_mag, rc_parallax)
        output = fakemag

    else:
        raise ValueError("Unknown unit")

    return ra, dec, output


def load_apokasc(combine=True):
    """
    Load APOKASC asteroseismic surface gravity measurement

    :param combine: True to combine gold and basic standard, otherwise only get gold standard log(g)
    :type combine: boolean
    :return: numpy array of ra, dec, array
    :rtype: ndarrays
    :History:
        | 2017-Dec-23 - Written - Henry Leung (University of Toronto)
    """
    catalog_list = Vizier.find_catalogs("apokasc")
    Vizier.ROW_LIMIT = 99999
    catalogs_gold = Vizier.get_catalogs(catalog_list.keys())[1]
    catalogs_basic = Vizier.get_catalogs(catalog_list.keys())[2]
    gold_ra = catalogs_gold["_RA"]
    gold_dec = catalogs_gold["_DE"]
    gold_logg = catalogs_gold["log_g_"]
    basic_ra = catalogs_basic["_RA"]
    basic_dec = catalogs_basic["_DE"]
    basic_logg = catalogs_basic["log.g2"]

    if combine is True:
        ra = np.append(np.array(gold_ra), np.array(basic_ra))
        dec = np.append(np.array(gold_dec), np.array(basic_dec))
        logg = np.append(np.array(gold_logg), np.array(basic_logg))
        return ra, dec, logg
    else:
        return gold_ra, gold_dec, gold_logg, basic_ra, basic_dec, basic_logg
