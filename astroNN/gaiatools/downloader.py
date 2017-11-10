# ---------------------------------------------------------#
#   astroNN.gaiatools.downloader: download gaia files
# ---------------------------------------------------------#

import os
import urllib.request

from astroNN.shared.downloader_tools import TqdmUpTo

currentdir = os.getcwd()

_GAIA_DATA = os.getenv('GAIA_TOOLS_DATA')


def tgas(dr=None):
    """
    NAME: tgas
    PURPOSE: download the tgas files
    INPUT:
    OUTPUT: (just downloads)
    HISTORY:
        2017-Oct-13 Henry Leung
    """

    # Check if dr arguement is provided, if none then use default
    if dr is None:
        dr = 1
        print('dr is not provided, using default dr=1')

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
                print('Downloaded Gaia DR{:d} TGAS ({:d} of 15) file catalog successfully to {}'.format(dr, i,
                                                                                                        fullfilename))
            else:
                print(fullfilename + ' was found, not downloaded again')
    else:
        raise ValueError('[astroNN.gaiatools.downloader.tgas()] only supports Gaia DR1 TGAS')

    return None


def gaia_source(dr=None):
    # TODO not working
    """
    NAME: gaia_source
    PURPOSE: download the gaia_source files
    INPUT:
    OUTPUT: (just downloads)
    HISTORY:
        2017-Oct-13 Henry Leung
    """
    if dr is None:
        dr = 1
        print('dr is not provided, using default dr=1')

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
        raise ValueError('[astroNN.gaiatools.downloader.gaia_source()] only supports Gaia DR1 Gaia Source')

    return None
