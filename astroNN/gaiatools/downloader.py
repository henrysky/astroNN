# ---------------------------------------------------------#
#   astroNN.gaiatools.downloader: download gaia files
# ---------------------------------------------------------#

import urllib.request
import os
from tqdm import tqdm

currentdir = os.getcwd()


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

    if dr == 1:
        # Check if directory exists
        if not os.path.exists(os.path.join(currentdir, 'TGAS\\')):
            os.makedirs(os.path.join(currentdir, 'TGAS\\'))

        for i in range(0, 16, 1):
            filename = 'TgasSource_000-000-0{:02d}.fits'.format(i)
            fullfilename = os.path.join(currentdir, 'TGAS\\', filename)
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

    if dr == 1:
        for i in range(0, 16, 1):
            url = 'http://cdn.gea.esac.esa.int/Gaia/gaia_source/fits/GaiaSource_000-000-0{:02d}.fits'.format(i)
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
                urllib.request.urlretrieve(url, reporthook=t.update_to)
            print('Downloaded Gaia DR{:d} Gaia Source ({:d} of {:d}) file catalog successfully to {}')% (dr, i, max(i),
                                                                                                      currentdir)
    else:
        raise ValueError('[astroNN.gaiatools.downloader.gaia_source()] only supports Gaia DR1 Gaia Source')

    return None