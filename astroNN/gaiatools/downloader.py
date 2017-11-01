# ---------------------------------------------------------#
#   astroNN.gaiatools.downloader: download gaia files
# ---------------------------------------------------------#

import urllib.request
import os
from tqdm import tqdm

currentdir = os.getcwd()

_APOGEE_DATA= os.getenv('SDSS_LOCAL_SAS_MIRROR')


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


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
        if not os.path.exists(os.path.join(currentdir, 'TGAS/')):
            os.makedirs(os.path.join(currentdir, 'TGAS/'))

        for i in range(0, 16, 1):
            filename = 'TgasSource_000-000-0{:02d}.fits'.format(i)
            fullfilename = os.path.join(currentdir, 'TGAS/', filename)
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
                urlstr = 'http://cdn.gea.esac.esa.int/Gaia/gaia_source/fits/GaiaSource_000-0{:02d}-{:03d}.fits'.format(j, i)
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=urlstr.split('/')[-1]) as t:
                    urllib.request.urlretrieve(urlstr, reporthook=t.update_to)
                print('Downloaded Gaia DR{:d} Gaia Source ({:d} of {:d}) file catalog successfully to {}') % (
                dr, (j*256 + i), 256*20 + 112, currentdir)
        for i in range(0, 111, 1):
            urlstr = 'http://cdn.gea.esac.esa.int/Gaia/gaia_source/fits/GaiaSource_000-020-{:03d}.fits'.format(i)
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=urlstr.split('/')[-1]) as t:
                urllib.request.urlretrieve(urlstr, reporthook=t.update_to)
            print('Downloaded Gaia DR{:d} Gaia Source ({:d} of {:d}) file catalog successfully to {}') % (
                dr, (20*256 + i), 256*20 + 112, currentdir)
    else:
        raise ValueError('[astroNN.gaiatools.downloader.gaia_source()] only supports Gaia DR1 Gaia Source')

    return None