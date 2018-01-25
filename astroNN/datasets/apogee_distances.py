# ---------------------------------------------------------#
#   astroNN.datasets.apogee_distances: APOGEE Distances
# ---------------------------------------------------------#

from astropy.io import fits

from astroNN.apogee.downloader import apogee_distances


def apogee_distances_load(dr=None):
    """
    NAME:
        apogee_distances_load
    PURPOSE:
        load apogee distances (absolute magnitude from stellar model)
    INPUT:
    OUTPUT:
    HISTORY:
        2018-Jan-25 - Written - Henry Leung (University of Toronto)
    """
    fullfilename = apogee_distances(dr=dr, verbose=1)

    with fits.open(fullfilename) as F:
        hdulist = F[1].data
        distance = hdulist['BPG_dist50']
        dist_err = hdulist['BPG_dist84'] - hdulist['BPG_dist16']
        rc_parallax = 1 / (distance)  # Convert kpc to parallax

    return distance, dist_err