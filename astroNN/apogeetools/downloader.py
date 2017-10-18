# ---------------------------------------------------------#
#   astroNN.apogeetools.downloader: download apogee files
# ---------------------------------------------------------#

import urllib.request
from tqdm import tqdm
from astropy.io import fits
import os

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

    # Check if dr arguement is provided, if none then use default
    if dr is None:
        dr = 14
        print('dr is not provided, using default dr=14')

    if dr == 13:
        # Check if directory exists
        fullfilepath = os.path.join(currentdir, 'apogee_dr13\\')
        if not os.path.exists(fullfilepath):
            os.makedirs(fullfilepath)
        filename = 'allStar-l30e.2.fits'
        fullfilename = os.path.join(fullfilepath, filename)
        url = 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/{}'.format(filename)
    elif dr == 14:
        fullfilepath = os.path.join(currentdir, 'apogee_dr14\\')
        # Check if directory exists
        if not os.path.exists(fullfilepath):
            os.makedirs(fullfilepath)
        filename = 'allStar-l31c.2.fits'
        fullfilename = os.path.join(fullfilepath, filename)
        url = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/{}'.format(filename)
    else:
        raise ValueError('[astroNN.apogeetools.downloader.all_star()] only supports APOGEE DR13 and DR14')

    # Check if files exists
    if not os.path.isfile(os.path.join(fullfilepath, filename)):
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, reporthook=t.update_to)
            print('Downloaded DR{:d} allStar file catalog successfully to {}'.format(dr, fullfilename))
    else:
        print(fullfilename + ' was found, not downloaded again')

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
        print('dr is not provided, using default dr=14')

    if dr == 13:
        # Check if directory exists
        fullfilepath = os.path.join(currentdir, 'apogee_allvisited_dr13\\')
        if not os.path.exists(fullfilepath):
            os.makedirs(fullfilepath)
        filename = 'allVisit-l30e.2.fits'
        fullfilename = os.path.join(fullfilepath, filename)
        url = 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/{}'.format(filename)
    elif dr == 14:
        # Check if directory exists
        fullfilepath = os.path.join(currentdir, 'apogee_allvisited_dr14\\')
        if not os.path.exists(fullfilepath):
            os.makedirs(fullfilepath)
        filename = 'allVisit-l31c.2.fits'
        fullfilename = os.path.join(fullfilepath, filename)
        url = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/{}'.format(filename)
    else:
        raise ValueError('[astroNN.apogeetools.downloader.all_visit()] only supports APOGEE DR13 and DR14')

    if not os.path.isfile(os.path.join(fullfilepath, filename)):
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, reporthook=t.update_to)
            print('Downloaded DR{:d} allVisit file catalog successfully to {}'.format(dr, currentdir))
    else:
        print(fullfilename + ' was found, not downloaded again')

    return None


def combined_spectra(dr=None):
    """
    NAME: combined_spectra
    PURPOSE: download the combined spectra file (catalog of properties from individual v isit spectra)
    INPUT: Data Release 13 OR 14
    OUTPUT: (just downloads)
    HISTORY:
        2017-Oct-15 Henry Leung
    """
    if dr is None:
        dr = 14
        print('dr is not provided, using default dr=14')

    if dr == 13:
        allstarepath = os.path.join(currentdir, 'apogee_dr13\\allVisit-l30e.2.fits')
        # Check if directory exists
        if not os.path.exists(allstarepath):
            os.makedirs(allstarepath)
            print('allStar catalog not found, please use astroNN.apogeetools.downloader.all_star(dr=13) to download it')
        else:
            print('allStar catalog DR13 has found successfully, now loading it')

        hdulist = fits.open(allstarepath)
        apogee_id = hdulist[1].data['APOGEE_ID']
        location_id = hdulist[1].data['LOCATION_ID']

        totalfiles = sum(1 for entry in os.listdir(os.path.join(currentdir, 'apogee_dr14\\')) if
                         os.path.isfile(os.path.join(os.path.join(currentdir, 'apogee_dr14\\'), entry)))

        if totalfiles > 12000:
            check = False
        else:
            check = True

        if check is True:
            for i in range(len(apogee_id)):
                str1 = 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/'
                str2 = '{}/aspcapStar-r6-l30e.2-{}.fits'.format(location_id[i], apogee_id[i])
                filename = 'aspcapStar-r6-l30e.2-{}.fits'.format(apogee_id[i])
                urlstr = str1 + str2
                filepath = os.path.join(currentdir, 'apogee_dr13\\', filename)
                if check is True and not os.path.isfile(filepath):
                    try:
                        urllib.request.urlretrieve(urlstr, filepath)
                        print('Downloaded DR13 combined file successfully to {}'.format(filepath))
                    except urllib.request.HTTPError:
                        print('{} cannot be found on server, skipped'.format(urlstr))
                else:
                    print(filepath + ' was found, not downloaded again')
        else:
            print('All DR13 combined spectra were found, not downloaded again')

    elif dr == 14:
        allstarepath = os.path.join(currentdir, 'apogee_dr14\\allStar-l31c.2.fits')
        # Check if directory exists
        if not os.path.exists(allstarepath):
            os.makedirs(allstarepath)
            print('allStar catalog not found, please use astroNN.apogeetools.downloader.all_star(dr=14) to download it')
        else:
            print('allStar catalog DR13 has found successfully, now loading it')

        hdulist = fits.open(allstarepath)
        apogee_id = hdulist[1].data['APOGEE_ID']
        location_id = hdulist[1].data['LOCATION_ID']

        totalfiles = sum(1 for entry in os.listdir(os.path.join(currentdir, 'apogee_dr14\\')) if
                         os.path.isfile(os.path.join(os.path.join(currentdir, 'apogee_dr14\\'), entry)))

        if totalfiles > 249480:
            check = False
        else:
            check = True

        if check is True:
            for i in range(len(apogee_id)):
                str1 = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/'
                str2 = '{}/aspcapStar-r8-l31c.2-{}.fits'.format(location_id[i], apogee_id[i])
                filename = 'aspcapStar-r8-l31c.2-{}.fits'.format(apogee_id[i])
                urlstr = str1 + str2
                filepath = os.path.join(currentdir, 'apogee_dr14\\', filename)
                if not os.path.isfile(filepath):
                    try:
                        urllib.request.urlretrieve(urlstr, filepath)
                        print('Downloaded DR14 combined file successfully to {}'.format(filepath))
                    except urllib.request.HTTPError:
                        print('{} cannot be found on server, skipped'.format(urlstr))
                else:
                    print(filepath + ' was found, not downloaded again')
            else:
                print('All DR14 combined spectra  were found, not downloaded again')

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
        print('dr is not provided, using default dr=14')
    return None
