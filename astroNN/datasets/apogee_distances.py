# ---------------------------------------------------------#
#   astroNN.datasets.apogee_distances: APOGEE Distances
# ---------------------------------------------------------#

import numpy as np
from astroNN.apogee import allstar
from astroNN.apogee.downloader import apogee_distances
from astroNN.gaia import mag_to_absmag, mag_to_fakemag, extinction_correction
from astropy import units as u
from astropy.io import fits


# noinspection PyUnresolvedReferences
def load_apogee_distances(dr=None, metric='distance', cuts=True, extinction=False):
    """
    Load apogee distances (absolute magnitude from stellar model)

    :param dr: Apogee DR
    :type dr: int
    :param metric: which metric you want ot get back

                   - "absmag" for absolute magnitude
                   - "fakemag" for fake magnitude
                   - "distance" for distance
    :type metric: string
    :param cuts: Whether to cut bad data (negative parallax and percentage error more than 20%), or a float to set the threshold
    :type cuts: Union[boolean, float]
    :param extinction: Whether to take extinction into account
    :type extinction: bool
    :return: numpy array of ra, dec, metrics_array, metrics_err_array
    :rtype: ndarrays
    :History: 2018-Jan-25 - Written - Henry Leung (University of Toronto)
    """
    fullfilename = apogee_distances(dr=dr)

    with fits.open(fullfilename) as F:
        hdulist = F[1].data
        # Convert kpc to pc
        distance = hdulist['BPG_dist50'] * 1000
        dist_err = (hdulist['BPG_dist84'] - hdulist['BPG_dist16']) * 1000

    allstarfullpath = allstar(dr=dr)

    with fits.open(allstarfullpath) as F:
        k_mag = F[1].data['K']
        if extinction:
            k_mag = extinction_correction(k_mag, F[1].data['AK_TARG'])
        ra = F[1].data['RA']
        dec = F[1].data['DEC']

    # Bad index refers to nan index
    bad_index = np.argwhere(np.isnan(distance))

    if metric == 'distance':
        # removed astropy units because of -9999. is dimensionless, will have issues
        output = distance
        output_err = dist_err

    elif metric == 'absmag':
        absmag, absmag_err = mag_to_absmag(k_mag, 1 / distance * u.arcsec, (1 / distance) * (dist_err / distance))
        output = absmag
        output_err = absmag_err

    elif metric == 'fakemag':
        # fakemag requires parallax (mas)
        fakemag, fakemag_err = mag_to_fakemag(k_mag, 1000 / distance * u.mas, (1000 / distance) * (dist_err / distance))
        output = fakemag
        output_err = fakemag_err

    else:
        raise ValueError('Unknown metric')

    # Set the nan index to -9999. as they are bad and unknown. Not magic_number as this is an APOGEE dataset
    output[bad_index], output_err[bad_index] = -9999., -9999.
    if cuts is False:
        pass
    else:
        distance[bad_index], dist_err[bad_index] = -9999., -9999.
        good_idx = [(dist_err / distance < (0.2 if cuts is True else cuts)) & (distance != -9999.)]

        ra = ra[good_idx]
        dec = dec[good_idx]
        output = output[good_idx]
        output_err = output_err[good_idx]

    return ra, dec, output, output_err
