# ---------------------------------------------------------#
#   astroNN.datasets.apogee_rc: APOGEE RC
# ---------------------------------------------------------#

import os

from astropy import units as u
from astropy.io import fits

from astroNN.apogee.downloader import apogee_vac_rc
from astroNN.gaia.gaia_shared import mag_to_absmag, mag_to_fakemag


def load_apogee_rc(dr=None, metric='distance'):
    """
    NAME:
        load_apogee_rc
    PURPOSE:
        load apogee red clumps (absolute magnitude measurement)
    INPUT:
    OUTPUT:
    HISTORY:
        2018-Jan-21 - Written - Henry Leung (University of Toronto)
    """
    fullfilename = apogee_vac_rc(dr=dr, verbose=1)

    with fits.open(fullfilename) as F:
        hdulist = F[1].data
        RA = hdulist['RA']
        DEC = hdulist['DEC']
        rc_dist = hdulist['RC_DIST']
        rc_parallax = (1 / rc_dist) * u.mas  # Convert kpc to parallax in mas
        K_mag = hdulist['K']

    if metric == 'distance':
        output = rc_dist * 1000 * u.parsec

    elif metric == 'absmag':
        absmag = mag_to_absmag(K_mag, 1 / (1000 * rc_dist) * u.arcsec)
        output = absmag

    elif metric == 'fakemag':
        # fakemag requires parallax (mas)
        fakemag = mag_to_fakemag(K_mag, 1 / rc_dist * u.mas)
        output = fakemag

    else:
        raise ValueError('Unknown metric')

    return RA, DEC, output
