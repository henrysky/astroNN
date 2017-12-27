# ---------------------------------------------------------#
#   astroNN.apogee.downloader: download apogee files
# ---------------------------------------------------------#

import os
import urllib.request

from astroNN.apogee.apogee_shared import apogee_env, apogee_default_dr
from astroNN.shared.downloader_tools import TqdmUpTo

currentdir = os.getcwd()

_APOGEE_DATA = apogee_env()


def allstar(dr=None):
    """
    NAME:
        allstar
    PURPOSE:
        download the allStar file (catalog of ASPCAP stellar parameters and abundances from combined spectra)
    INPUT:
        dr (int): APOGEE DR, example dr=14
    OUTPUT:
        (path): full file path and download in background
    HISTORY:
        2017-Oct-09 - Written - Henry Leung (University of Toronto)
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
    NAME:
        allstarcanon
    PURPOSE:
        download the allStarCannon file (catalog of Cannon stellar parameters and abundances from combined spectra)
    INPUT:
        dr (int): APOGEE DR, example dr=14
    OUTPUT:
        (path): full file path and download in background
    HISTORY:
        2017-Oct-24 - Written - Henry Leung (University of Toronto)
    """

    dr = apogee_default_dr(dr=dr)

    if dr == 14:
        pass
    elif dr == 13:
        print('allstarcanon() currently not supporting DR13')
    else:
        raise ValueError('[astroNN.apogee.downloader.allstarcannon() ] only supports APOGEE DR14')

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
    NAME:
        allvisit
    PURPOSE:
        download the allVisit file (catalog of properties from individual visit spectra)
    INPUT:
        dr (int): APOGEE DR, example dr=14
    OUTPUT:
        None, (just downloads)
    HISTORY:
        2017-Oct-11 - Written - Henry Leung (University of Toronto)
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


def combined_spectra(dr=None, location=None, apogee=None, verbose=1):
    """
    NAME:
        combined_spectra
    PURPOSE:
        download the required combined spectra file (catalog of properties from individual visit spectra)
    INPUT:
        dr (int): APOGEE DR, example dr=14
    OUTPUT:
        None, (just downloads)
    HISTORY:
        2017-Oct-15 - Written - Henry Leung (University of Toronto)
    """
    warning_flag = None

    dr = apogee_default_dr(dr=dr)

    if dr == 13:
        str1 = 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/'
        str2 = '{}/aspcapStar-r6-l30e.2-{}.fits'.format(location, apogee)
        filename = 'aspcapStar-r6-l30e.2-{}.fits'.format(apogee)
        urlstr = str1 + str2
        fullfilename = os.path.join(_APOGEE_DATA, 'dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/', str(location))
        if not os.path.exists(fullfilename):
            os.makedirs(fullfilename)
        fullfilename = os.path.join(_APOGEE_DATA, 'dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/', str(location),
                                    filename)
        if not os.path.isfile(fullfilename):
            try:
                urllib.request.urlretrieve(urlstr, fullfilename)
                print('Downloaded DR13 combined file successfully to {}'.format(fullfilename))
            except urllib.request.HTTPError:
                print('{} cannot be found on server, skipped'.format(urlstr))
        else:
            print(fullfilename + ' was found, not downloaded again')

    elif dr == 14:
        str1 = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/'
        str2 = '{}/aspcapStar-r8-l31c.2-{}.fits'.format(location, apogee)
        filename = 'aspcapStar-r8-l31c.2-{}.fits'.format(apogee)
        urlstr = str1 + str2
        fullfilename = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/', str(location))
        if not os.path.exists(fullfilename):
            os.makedirs(fullfilename)
        fullfilename = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/', str(location),
                                    filename)
        if not os.path.isfile(fullfilename):
            try:
                urllib.request.urlretrieve(urlstr, fullfilename)
                print('Downloaded DR14 combined file successfully to {}'.format(fullfilename))
            except urllib.request.HTTPError:
                print('{} cannot be found on server, skipped'.format(urlstr))
                warning_flag = 1
        else:
            if verbose == 1:
                print(fullfilename + ' was found, not downloaded again')

    else:
        raise ValueError('combined_spectra() only supports DR13 or DR14')

    return warning_flag, fullfilename


def visit_spectra(dr=None, location=None, apogee=None, verbose=1):
    """
    NAME:
        visit_spectra
    PURPOSE:
        download the combined spectra file (catalog of properties from individual visit spectra)
    INPUT:
        dr (int): APOGEE DR, example dr=14
    OUTPUT:
        None, (just downloads)
    HISTORY:
        2017-Nov-11 - Written - Henry Leung (University of Toronto)
    """
    warning_flag = None

    dr = apogee_default_dr(dr=dr)

    if dr == 13:
        str1 = 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/apo25m/'
        str2 = '{}/apStar-r6-{}.fits'.format(location, apogee)
        filename = 'apStar-r6-{}.fits'.format(apogee)
        urlstr = str1 + str2
        fullfilename = os.path.join(_APOGEE_DATA, 'dr13/apogee/spectro/redux/r6/stars/apo25m/', str(location))
        if not os.path.exists(fullfilename):
            os.makedirs(fullfilename)
        fullfilename = os.path.join(_APOGEE_DATA, 'dr13/apogee/spectro/redux/r6/stars/apo25m/', str(location), filename)
        if not os.path.isfile(fullfilename):
            try:
                urllib.request.urlretrieve(urlstr, fullfilename)
                print('Downloaded DR13 individual visit file successfully to {}'.format(fullfilename))
            except urllib.request.HTTPError:
                print('{} cannot be found on server, skipped'.format(urlstr))
        else:
            print(fullfilename + ' was found, not downloaded again')

    elif dr == 14:
        str1 = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/apo25m/'
        str2 = '{}/apStar-r8-{}.fits'.format(location, apogee)
        filename = 'apStar-r8-{}.fits'.format(apogee)
        urlstr = str1 + str2
        fullfilename = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/apo25m/', str(location))
        if not os.path.exists(fullfilename):
            os.makedirs(fullfilename)
        fullfilename = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/apo25m/', str(location), filename)
        if not os.path.isfile(fullfilename):
            try:
                urllib.request.urlretrieve(urlstr, fullfilename)
                print('Downloaded DR14 individual visit file successfully to {}'.format(fullfilename))
            except urllib.request.HTTPError:
                print('{} cannot be found on server, skipped'.format(urlstr))
                warning_flag = 1
        else:
            if verbose == 1:
                print(fullfilename + ' was found, not downloaded again')

    else:
        raise ValueError('visit_spectra() only supports DR13 or DR14')

    return warning_flag, fullfilename


def apogee_vac_rc(dr=None, verbose=1):
    """
    NAME:
        apogee_vac_rc
    PURPOSE:
        download the red clumps catalogue
    INPUT:
        dr (int): APOGEE DR, example dr=14
    OUTPUT:
        None, (just downloads)
    HISTORY:
        2017-Nov-16 - Written - Henry Leung (University of Toronto)
    """
    warning_flag = None

    dr = apogee_default_dr(dr=dr)

    if dr == 13:
        str1 = 'https://data.sdss.org/sas/dr13/apogee/vac/apogee-rc/cat/'
        filename = 'apogee-rc-DR{}.fits'.format(dr)
        urlstr = str1 + filename
        fullfilename = os.path.join(_APOGEE_DATA, 'dr13/apogee/vac/apogee-rc/cat/')
        if not os.path.exists(fullfilename):
            os.makedirs(fullfilename)
        fullfilename = os.path.join(_APOGEE_DATA, 'dr13/apogee/vac/apogee-rc/cat/', filename)
        if not os.path.isfile(fullfilename):
            try:
                urllib.request.urlretrieve(urlstr, fullfilename)
                print('Downloaded DR13 Red Clumps file successfully to {}'.format(fullfilename))
            except urllib.request.HTTPError:
                print('{} cannot be found on server, skipped'.format(urlstr))
        else:
            print(fullfilename + ' was found, not downloaded again')

    elif dr == 14:
        str1 = 'https://data.sdss.org/sas/dr14/apogee/vac/apogee-rc/cat/'
        filename = 'apogee-rc-DR{}.fits'.format(dr)
        urlstr = str1 + filename
        fullfilename = os.path.join(_APOGEE_DATA, 'dr14/apogee/vac/apogee-rc/cat/')
        if not os.path.exists(fullfilename):
            os.makedirs(fullfilename)
        fullfilename = os.path.join(_APOGEE_DATA, 'dr14/apogee/vac/apogee-rc/cat/', filename)
        if not os.path.isfile(fullfilename):
            try:
                urllib.request.urlretrieve(urlstr, fullfilename)
                print('Downloaded DR14 Red Clumps file successfully to {}'.format(fullfilename))
            except urllib.request.HTTPError:
                print('{} cannot be found on server, skipped'.format(urlstr))
                warning_flag = 1
        else:
            if verbose == 1:
                print(fullfilename + ' was found, not downloaded again')

    else:
        raise ValueError('apogee_vac_rc() only supports DR13 or DR14')

    return warning_flag, fullfilename
