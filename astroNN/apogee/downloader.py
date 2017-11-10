# ---------------------------------------------------------#
#   astroNN.apogee.downloader: download apogee files
# ---------------------------------------------------------#

import os
import urllib.request

from astropy.io import fits

from astroNN.shared.downloader_tools import TqdmUpTo
from astroNN.apogee.apogee_shared import apogee_env, apogee_default_dr

currentdir = os.getcwd()

_APOGEE_DATA = apogee_env()


def allstar(dr=None):
    """
    NAME: allstar
    PURPOSE: download the allStar file (catalog of ASPCAP stellar parameters and abundances from combined spectra)
    INPUT: Data Release 13 OR 14
    OUTPUT: full file path and download in background
    HISTORY:
        2017-Oct-09 Henry Leung
    """

    dr = apogee_default_dr(dr=dr)

    if dr == 13:
        # Check if directory exists
        fullfilepath = os.path.join(_APOGEE_DATA, 'dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/')
        if not os.path.exists(fullfilepath):
            os.makedirs(fullfilepath)
        filename = 'allStar-l30e.2.fits'
        fullfilename = os.path.join(fullfilepath, filename)
        url = 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/{}'.format(filename)
    elif dr == 14:
        fullfilepath = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/')
        # Check if directory exists
        if not os.path.exists(fullfilepath):
            os.makedirs(fullfilepath)
        filename = 'allStar-l31c.2.fits'
        fullfilename = os.path.join(fullfilepath, filename)
        url = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/{}'.format(filename)
    else:
        raise ValueError('[astroNN.apogee.downloader.all_star()] only supports APOGEE DR13 and DR14')

    # Check if files exists
    if not os.path.isfile(os.path.join(fullfilepath, filename)):
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, fullfilename, reporthook=t.update_to)
            print('Downloaded DR{:d} allStar file catalog successfully to {}'.format(dr, fullfilename))
    else:
        print(fullfilename + ' was found!')

    return fullfilename


def allstarcannon(dr=None):
    """
    NAME: allstarcanon
    PURPOSE: download the allStarCannon file (catalog of Cannon stellar parameters and abundances from combined spectra)
    INPUT: Data Release 14
    OUTPUT: full file path and download in background
    HISTORY:
        2017-Oct-24 Henry Leung
    """

    dr = apogee_default_dr(dr=dr)

    if dr == 14:
        pass
    elif dr == 13:
        print('allstarcanon() currently not supporting DR13')
    else:
        raise ValueError('[astroNN.apogee.downloader.allstarcannon()] only supports APOGEE DR14')

    # Check if directory exists
    fullfilepath = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/cannon/')
    # Check if directory exists
    if not os.path.exists(fullfilepath):
        os.makedirs(fullfilepath)
    filename = 'allStarCannon-l31c.2.fits'
    fullfilename = os.path.join(fullfilepath, filename)
    url = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/cannon/{}'.format(filename)

    # Check if files exists
    if not os.path.isfile(os.path.join(fullfilepath, filename)):
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, fullfilename, reporthook=t.update_to)
            print('Downloaded DR{:d} allStarCannon file catalog successfully to {}'.format(dr, fullfilename))
    else:
        print(fullfilename + ' was found')

    return fullfilename


def allvisit(dr=None):
    """
    NAME: allvisit
    PURPOSE: download the allVisit file (catalog of properties from individual visit spectra)
    INPUT: Data Release 13 OR 14
    OUTPUT: (just downloads)
    HISTORY:
        2017-Oct-11 Henry Leung
    """

    dr = apogee_default_dr(dr=dr)

    if dr == 13:
        # Check if directory exists
        fullfilepath = os.path.join(_APOGEE_DATA, 'dr13/apogee/spectro/redux/r6/')
        if not os.path.exists(fullfilepath):
            os.makedirs(fullfilepath)
        filename = 'allVisit-l30e.2.fits'
        fullfilename = os.path.join(fullfilepath, filename)
        url = 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/{}'.format(filename)
    elif dr == 14:
        # Check if directory exists
        fullfilepath = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/')
        if not os.path.exists(fullfilepath):
            os.makedirs(fullfilepath)
        filename = 'allVisit-l31c.2.fits'
        fullfilename = os.path.join(fullfilepath, filename)
        url = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/{}'.format(filename)
    else:
        raise ValueError('[astroNN.apogee.downloader.all_visit()] only supports APOGEE DR13 and DR14')

    if not os.path.isfile(os.path.join(fullfilepath, filename)):
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, fullfilename, reporthook=t.update_to)
            print('Downloaded DR{:d} allVisit file catalog successfully to {}'.format(dr, fullfilepath))
    else:
        print(fullfilename + ' was found')

    return None


def combined_spectra(dr=None, download_all=False, location=None, apogee=None):
    """
    NAME: combined_spectra
    PURPOSE: download the required combined spectra file (catalog of properties from individual visit spectra)
    INPUT: Data Release 13 OR 14
    OUTPUT: (just downloads)
    HISTORY:
        2017-Oct-15 Henry Leung
    """
    warning_flag = None

    dr = apogee_default_dr(dr=dr)

    if dr == 13 and download_all is True:
        allstarpath = allstar(dr=13)
        hdulist = fits.open(allstarpath)
        apogee_id = hdulist[1].data['APOGEE_ID']
        location_id = hdulist[1].data['LOCATION_ID']

        totalfiles = sum(1 for entry in os.listdir(os.path.join(currentdir, 'apogee_dr14/')) if
                         os.path.isfile(os.path.join(os.path.join(currentdir, 'apogee_dr14/'), entry)))

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
                filepath = os.path.join(currentdir, 'apogee_dr13/', filename)
                if check is True and not os.path.isfile(filepath):
                    try:
                        urllib.request.urlretrieve(urlstr, filepath)
                        print('Downloaded DR13 combined file successfully to {}'.format(filepath))
                    except urllib.request.HTTPError:
                        print('{} cannot be found on server, skipped'.format(urlstr))
                        warning_flag = 1
                else:
                    print(filepath + ' was found, not downloaded again')
        else:
            print('All DR13 combined spectra were found, not downloaded again')

    elif dr == 13 and download_all is False:
        str1 = 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/'
        str2 = '{}/aspcapStar-r6-l30e.2-{}.fits'.format(location, apogee)
        filename = 'aspcapStar-r6-l30e.2-{}.fits'.format(apogee)
        urlstr = str1 + str2
        filepath = os.path.join(_APOGEE_DATA, 'dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/', str(location),
                                filename)
        if not os.path.isfile(filepath):
            try:
                urllib.request.urlretrieve(urlstr, filepath)
                print('Downloaded DR13 combined file successfully to {}'.format(filepath))
            except urllib.request.HTTPError:
                print('{} cannot be found on server, skipped'.format(urlstr))
        else:
            print(filepath + ' was found, not downloaded again')

    elif dr == 14 and download_all is True:
        allstarpath = allstar(dr=14)
        hdulist = fits.open(allstarpath)
        apogee_id = hdulist[1].data['APOGEE_ID']
        location_id = hdulist[1].data['LOCATION_ID']

        totalfiles = sum(1 for entry in os.listdir(os.path.join(currentdir, 'apogee_dr14/')) if
                         os.path.isfile(os.path.join(os.path.join(currentdir, 'apogee_dr14/'), entry)))

        if totalfiles > 263062:
            check = False
        else:
            check = True

        if check is True:
            for i in range(len(apogee_id)):
                str1 = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/'
                str2 = '{}/aspcapStar-r8-l31c.2-{}.fits'.format(location_id[i], apogee_id[i])
                filename = 'aspcapStar-r8-l31c.2-{}.fits'.format(apogee_id[i])
                urlstr = str1 + str2
                filepath = os.path.join(currentdir, 'apogee_dr14/', filename)
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

    elif dr == 14 and download_all is False:
        str1 = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/'
        str2 = '{}/aspcapStar-r8-l31c.2-{}.fits'.format(location, apogee)
        filename = 'aspcapStar-r8-l31c.2-{}.fits'.format(apogee)
        urlstr = str1 + str2
        filepath = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/', str(location))
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filepath = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/', str(location),
                                filename)
        if not os.path.isfile(filepath):
            try:
                urllib.request.urlretrieve(urlstr, filepath)
                print('Downloaded DR14 combined file successfully to {}'.format(filepath))
            except urllib.request.HTTPError:
                print('{} cannot be found on server, skipped'.format(urlstr))
                warning_flag = 1
        else:
            print(filepath + ' was found, not downloaded again')

    return warning_flag


def visit_spectra(dr=None):
    """
    NAME: visit_spectra
    PURPOSE: download the combined spectra file (catalog of properties from individual visit spectra)
    INPUT: Data Release 13 OR 14
    OUTPUT: (just downloads)
    HISTORY:
        2017-Oct-11 Henry Leung
    """
    dr = apogee_default_dr(dr=dr)

    return None
