# ---------------------------------------------------------#
#   astroNN.apogeetools.downloader: download apogee files
# ---------------------------------------------------------#

import urllib.request
import sys
import time
import os
from tqdm import tqdm

DR13_URL = 'http://data.sdss.org/sas/dr13'
DR14_URL = 'http://data.sdss.org/sas/dr14'

_DR13REDUX = 'l30e.2'
_DR14REDUX = 'l31c.2'

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

def allstar(dr=None):
    """
    NAME: all_star
    PURPOSE: download the allStar file (catalog of ASPCAP stellar parameters and abundances	from combined spectra)
    INPUT: Data Release 13 OR 14
    OUTPUT: (just downloads)
    HISTORY:
        2017-Oct-09 Henry Leung
    """
    if dr is None:
        dr = 14

    if dr == 13:
        url = 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/allStar-l30e.2.fits'
    elif dr == 14:
        url = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/allStar-l31c.2.fits'
    else:
        raise ValueError('[astroNN.apogeetools.downloader.all_star()] only supports DR13 and DR14 APOGEE')

    with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, reporthook=t.update_to)
    print('Downloaded DR{:d} allStar file catalog successfulliy to {}'.format(dr, currentdir))

    return None


def allvisit(dr=None):
    """
    NAME: all_visit
    PURPOSE: download the allVisit file (catalog of properties from individual visit spectra)
    INPUT: Data Release 13 OR 14
    OUTPUT: (just downloads)
    HISTORY:
        2017-Oct-11 Henry Leung
    """
    if dr is None:
        dr = 14

    if dr == 13:
        url = 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/allVisit-l30e.2.fits'
    elif dr == 14:
        url = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/allVisit-l31c.2.fits'
    else:
        raise ValueError('[astroNN.apogeetools.downloader.all_visit()] only supports DR13 and DR14 APOGEE')

    with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, reporthook=t.update_to)
    print('Downloaded DR{:d} allVisit file catalog successfulliy to {}'.format(dr, currentdir))

    return None


def combined_spectra(dr=None):
    """
    NAME: combined_spectra
    PURPOSE: download the combined spectra file (catalog of properties from individual visit spectra)
    INPUT: Data Release 13 OR 14
    OUTPUT: (just downloads)
    HISTORY:
        2017-Oct-11 Henry Leung
    """
    if dr is None:
        dr = 14
    return None


def visit_spectra(dr=None):
    """
    NAME: visit_spectra
    PURPOSE: download the combined spectra file (catalog of properties from individual visit spectra)
    INPUT: Data Release 13 OR 14
    OUTPUT: (just downloads)
    HISTORY:
        2017-Oct-11 Henry Leung
    """
    if dr is None:
        dr = 14
    return None

