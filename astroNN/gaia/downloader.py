# ---------------------------------------------------------#
#   astroNN.gaia.downloader: download gaia files
# ---------------------------------------------------------#

import os
import urllib.request
import numpy as np
from astropy.io import fits

from astroNN.shared.downloader_tools import TqdmUpTo
from astroNN.gaia.gaia_shared import gaia_env, gaia_default_dr

currentdir = os.getcwd()
_GAIA_DATA = gaia_env()


def tgas(dr=None):
    """
    NAME:
        tgas
    PURPOSE:
        download the tgas files
    INPUT:
        dr (int): GAIA DR, example dr=1
    OUTPUT:
        None (just downloads)
    HISTORY:
        2017-Oct-13 - Written - Henry Leung (University of Toronto)
    """

    # Check if dr arguement is provided, if none then use default
    dr = gaia_default_dr(dr=dr)
    fulllist = []

    if dr == 1:
        # Check if directory exists
        folderpath =os.path.join(_GAIA_DATA, 'Gaia/tgas_source/fits/')
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)

        for i in range(0, 16, 1):
            filename = 'TgasSource_000-000-0{:02d}.fits'.format(i)
            fullfilename = os.path.join(folderpath, filename)
            urlstr = 'http://cdn.gea.esac.esa.int/Gaia/tgas_source/fits/{}'.format(filename)

            # Check if files exists
            if not os.path.isfile(fullfilename):
                # progress bar
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=urlstr.split('/')[-1]) as t:
                    # Download
                    urllib.request.urlretrieve(urlstr, fullfilename, reporthook=t.update_to)
                print('Downloaded Gaia DR{:d} TGAS ({:d} of 15) file catalog successfully to {}'.format(dr, i, fullfilename))
            else:
                print(fullfilename + ' was found!')

            fulllist.extend([fullfilename])
    else:
        raise ValueError('tgas() only supports Gaia DR1 TGAS')

    return fulllist


def tgas_load(dr=None):
    """
    NAME:
        tgas_load
    PURPOSE:
        to load useful parameters from multiple TGAS files
    INPUT:
        dr (int): GAIA DR, example dr=1
    OUTPUT:
    HISTORY:
        2017-Dec-17 - Written - Henry Leung (University of Toronto)
    """
    dr = gaia_default_dr(dr=dr)
    tgas_list = tgas(dr=dr)

    ra_gaia = np.array([])
    dec_gaia = np.array([])
    pmra_gaia = np.array([])
    pmdec_gaia = np.array([])
    parallax_gaia = np.array([])
    parallax_error_gaia = np.array([])
    g_band_gaia = np.array([])

    for i in tgas_list:
        gaia = fits.open(i)
        ra_gaia = np.concatenate((ra_gaia, gaia[1].data['RA']))
        dec_gaia = np.concatenate((dec_gaia, gaia[1].data['DEC']))
        pmra_gaia = np.concatenate((pmra_gaia, gaia[1].data['PMRA']))
        pmdec_gaia = np.concatenate((pmdec_gaia, gaia[1].data['PMDEC']))
        parallax_gaia = np.concatenate((parallax_gaia, gaia[1].data['parallax']))
        parallax_error_gaia = np.concatenate((parallax_error_gaia, gaia[1].data['parallax_error']))
        g_band_gaia = np.concatenate((g_band_gaia, gaia[1].data['phot_g_mean_mag']))
        gaia.close()

    return ra_gaia, dec_gaia, pmra_gaia, pmdec_gaia, parallax_gaia, parallax_error_gaia, g_band_gaia


def gaia_source(dr=None):
    """
    NAME:
        gaia_source
    PURPOSE:
        download the gaia_source files
    INPUT:
        dr (int): GAIA DR, example dr=1
    OUTPUT:
        None (just downloads)
    HISTORY:
        2017-Oct-13 - Written - Henry Leung (University of Toronto)
        2017-Nov-26 - Update - Henry Leung (University of Toronto)
    """

    print("Currently gaia_source isnt working properly")

    dr = gaia_default_dr(dr=dr)

    if dr == 1:
        for j in range(0, 20, 1):
            for i in range(0, 256, 1):
                urlstr = 'http://cdn.gea.esac.esa.int/Gaia/gaia_source/fits/GaiaSource_000-0{:02d}-{:03d}.fits'.format(
                    j, i)
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=urlstr.split('/')[-1]) as t:
                    urllib.request.urlretrieve(urlstr, reporthook=t.update_to)
                print('Downloaded Gaia DR{:d} Gaia Source ({:d} of {:d}) file catalog successfully to {}') % (
                    dr, (j * 256 + i), 256 * 20 + 112, currentdir)
        for i in range(0, 111, 1):
            urlstr = 'http://cdn.gea.esac.esa.int/Gaia/gaia_source/fits/GaiaSource_000-020-{:03d}.fits'.format(i)
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=urlstr.split('/')[-1]) as t:
                urllib.request.urlretrieve(urlstr, reporthook=t.update_to)
            print('Downloaded Gaia DR{:d} Gaia Source ({:d} of {:d}) file catalog successfully to {}') % (
                dr, (20 * 256 + i), 256 * 20 + 112, currentdir)
    else:
        raise ValueError('gaia_source() only supports Gaia DR1 Gaia Source')

    return None
