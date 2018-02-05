# ---------------------------------------------------------#
#   astroNN.datasets.apogee_distances: APOGEE Distances
# ---------------------------------------------------------#

import numpy as np
from astropy import units as u
from astropy.io import fits

from astroNN.apogee import allstar
from astroNN.apogee.downloader import apogee_distances
from astroNN.gaia import mag_to_absmag, mag_to_fakemag


def load_apogee_distances(dr=None, metric='distance'):
    """
    NAME:
        load_apogee_distances
    PURPOSE:
        load apogee distances (absolute magnitude from stellar model)
    INPUT:
        metric (string): which metric you want ot get back
                "absmag" for absolute magnitude
                "fakemag" for fake magnitude
                "distance" for distance
    OUTPUT:
    HISTORY:
        2018-Jan-25 - Written - Henry Leung (University of Toronto)
    """
    fullfilename = apogee_distances(dr=dr, verbose=1)

    with fits.open(fullfilename) as F:
        hdulist = F[1].data
        # Convert kpc to pc
        distance = hdulist['BPG_dist50'] * 1000
        dist_err = (hdulist['BPG_dist84'] - hdulist['BPG_dist16']) * 1000

    allstarfullpath = allstar(dr=dr)

    with fits.open(allstarfullpath) as F:
        K_mag = F[1].data['K']
        RA = F[1].data['RA']
        DEC = F[1].data['DEC']

    # Bad index refers to nan index
    bad_index = np.argwhere(np.isnan(distance))

    if metric == 'distance':
        # removed astropy units because of -9999. is dimensionless, will have issues
        output = distance
        output_err = dist_err

    elif metric == 'absmag':
        absmag = mag_to_absmag(K_mag, 1 / distance * u.arcsec)
        output = absmag
        output_err = dist_err
        print('Error array is wrong, dont use it, I am sorry')

    elif metric == 'fakemag':
        # fakemag requires parallax (mas)
        fakemag = mag_to_fakemag(K_mag, 1000 / distance * u.mas)
        output = fakemag
        output_err = dist_err
        print('Error array is wrong, dont use it, I am sorry')

    else:
        raise ValueError('Unknown metric')

    # Set the nan index to -9999. as they are bad and unknown. Not magic_number as this is an APOGEE dataset
    output[bad_index], output_err[bad_index] = -9999., -9999.

    return RA, DEC, output, output_err
