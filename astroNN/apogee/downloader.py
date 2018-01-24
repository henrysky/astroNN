# ---------------------------------------------------------#
#   astroNN.apogee.downloader: download apogee files
# ---------------------------------------------------------#

import os
import urllib.request

from astroNN.apogee.apogee_shared import apogee_env, apogee_default_dr
from astroNN.shared.downloader_tools import TqdmUpTo
from astroNN.shared.downloader_tools import sha1_checksum

currentdir = os.getcwd()

_APOGEE_DATA = apogee_env()


def allstar(dr=None, flag=None):
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
        file_hash = '1718723ada3018de94e1022cd57d4d950a74f91f'

        # Check if directory exists
        fullfoldername = os.path.join(_APOGEE_DATA, 'dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/')
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
        filename = 'allStar-l30e.2.fits'
        fullfilename = os.path.join(fullfoldername, filename)
        url = 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/{}'.format(filename)
    elif dr == 14:
        file_hash = 'a7e1801924661954da792e377ad54f412219b105'

        fullfoldername = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/')
        # Check if directory exists
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
        filename = 'allStar-l31c.2.fits'
        fullfilename = os.path.join(fullfoldername, filename)
        url = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/{}'.format(filename)
    else:
        raise ValueError('[astroNN.apogee.downloader.all_star()] only supports APOGEE DR13 and DR14')

    # check file integrity
    if os.path.isfile(fullfilename) and flag is None:
        checksum = sha1_checksum(fullfilename)
        if checksum != file_hash.lower():
            print('File corruption detected, astroNN attempting to download again')
            allstar(dr=dr, flag=1)
        else:
            print(fullfilename + ' was found!')

    # Check if files exists
    if not os.path.isfile(os.path.join(fullfoldername, filename)):
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, fullfilename, reporthook=t.update_to)
            print('Downloaded DR{:d} allStar file catalog successfully to {}'.format(dr, fullfilename))
            checksum = sha1_checksum(fullfilename)
            if checksum != file_hash.lower():
                print('File corruption detected, astroNN attempting to download again')
                allstar(dr=dr, flag=1)

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
    fullfoldername = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/cannon/')
    # Check if directory exists
    if not os.path.exists(fullfoldername):
        os.makedirs(fullfoldername)
    filename = 'allStarCannon-l31c.2.fits'
    fullfilename = os.path.join(fullfoldername, filename)
    url = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/cannon/{}'.format(filename)

    # Check if files exists
    if not os.path.isfile(os.path.join(fullfoldername, filename)):
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, fullfilename, reporthook=t.update_to)
            print('Downloaded DR{:d} allStarCannon file catalog successfully to {}'.format(dr, fullfilename))
    else:
        print(fullfilename + ' was found')

    return fullfilename


def allvisit(dr=None, flag=None):
    """
    NAME:
        allvisit
    PURPOSE:
        download the allVisit file (catalog of properties from individual visit spectra)
    INPUT:
        dr (int): APOGEE DR, example dr=14
    OUTPUT:
        (path): full file path and download in background
    HISTORY:
        2017-Oct-11 - Written - Henry Leung (University of Toronto)
    """

    dr = apogee_default_dr(dr=dr)

    if dr == 13:
        file_hash = '2a3b13ccd40a2c8aea8321be9630117922d55b51'

        # Check if directory exists
        fullfilepath = os.path.join(_APOGEE_DATA, 'dr13/apogee/spectro/redux/r6/')
        if not os.path.exists(fullfilepath):
            os.makedirs(fullfilepath)
        filename = 'allVisit-l30e.2.fits'
        fullfilename = os.path.join(fullfilepath, filename)
        url = 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/{}'.format(filename)
    elif dr == 14:
        file_hash = 'abcecbcdc5fe8d00779738702c115633811e6bbd'

        # Check if directory exists
        fullfilepath = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/')
        if not os.path.exists(fullfilepath):
            os.makedirs(fullfilepath)
        filename = 'allVisit-l31c.2.fits'
        fullfilename = os.path.join(fullfilepath, filename)
        url = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/{}'.format(filename)
    else:
        raise ValueError('[astroNN.apogee.downloader.all_visit()] only supports APOGEE DR13 and DR14')

    # check file integrity
    if os.path.isfile(fullfilename) and flag is None:
        checksum = sha1_checksum(fullfilename)
        if checksum != file_hash.lower():
            print('File corruption detected, astroNN attempting to download again')
            allvisit(dr=dr, flag=1)
        else:
            print(fullfilename + ' was found!')
    elif not os.path.isfile(os.path.join(fullfilepath, filename)) or flag == 1:
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, fullfilename, reporthook=t.update_to)
            print('Downloaded DR{:d} allVisit file catalog successfully to {}'.format(dr, fullfilepath))
            checksum = sha1_checksum(fullfilename)
            if checksum != file_hash.lower():
                print('File corruption detected, astroNN attempting to download again')
                allstar(dr=dr, flag=1)

    return fullfilename


def combined_spectra(dr=None, location=None, apogee=None, verbose=1):
    """
    NAME:
        combined_spectra
    PURPOSE:
        download the required combined spectra file (catalog of properties from individual visit spectra)
    INPUT:
        dr (int): APOGEE DR, example dr=14
    OUTPUT:
        (path): full file path and download in background
    HISTORY:
        2017-Oct-15 - Written - Henry Leung (University of Toronto)
    """
    warning_flag = False

    dr = apogee_default_dr(dr=dr)

    if dr == 13:
        str1 = 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/'
        str2 = '{}/aspcapStar-r6-l30e.2-{}.fits'.format(location, apogee)
        filename = 'aspcapStar-r6-l30e.2-{}.fits'.format(apogee)
        urlstr = str1 + str2
        fullfoldername = os.path.join(_APOGEE_DATA, 'dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/', str(location))
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
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
        fullfoldername = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/', str(location))
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
        fullfilename = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/', str(location),
                                    filename)
        if not os.path.isfile(fullfilename):
            try:
                urllib.request.urlretrieve(urlstr, fullfilename)
                print('Downloaded DR14 combined file successfully to {}'.format(fullfilename))
            except urllib.request.HTTPError:
                print('{} cannot be found on server, skipped'.format(urlstr))
                fullfilename = warning_flag
        else:
            if verbose == 1:
                print(fullfilename + ' was found, not downloaded again')

    else:
        raise ValueError('combined_spectra() only supports DR13 or DR14')

    return fullfilename


def visit_spectra(dr=None, location=None, apogee=None, verbose=1):
    """
    NAME:
        visit_spectra
    PURPOSE:
        download the combined spectra file (catalog of properties from individual visit spectra)
    INPUT:
        dr (int): APOGEE DR, example dr=14
    OUTPUT:
        (path): full file path and download in background
    HISTORY:
        2017-Nov-11 - Written - Henry Leung (University of Toronto)
    """
    warning_flag = False

    dr = apogee_default_dr(dr=dr)

    if dr == 13:
        str1 = 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/apo25m/'
        str2 = '{}/apStar-r6-{}.fits'.format(location, apogee)
        filename = 'apStar-r6-{}.fits'.format(apogee)
        urlstr = str1 + str2
        fullfoldername = os.path.join(_APOGEE_DATA, 'dr13/apogee/spectro/redux/r6/stars/apo25m/', str(location))
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
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
        fullfoldername = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/apo25m/', str(location))
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
        fullfilename = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/apo25m/', str(location), filename)
        if not os.path.isfile(fullfilename):
            try:
                urllib.request.urlretrieve(urlstr, fullfilename)
                print('Downloaded DR14 individual visit file successfully to {}'.format(fullfilename))
            except urllib.request.HTTPError:
                print('{} cannot be found on server, skipped'.format(urlstr))
                fullfilename = warning_flag
        else:
            if verbose == 1:
                print(fullfilename + ' was found, not downloaded again')

    else:
        raise ValueError('visit_spectra() only supports DR13 or DR14')

    return fullfilename


def apogee_vac_rc(dr=None, verbose=1, flag=None):
    """
    NAME:
        apogee_vac_rc
    PURPOSE:
        download the red clumps catalogue
    INPUT:
        dr (int): APOGEE DR, example dr=14
    OUTPUT:
        (path): full file path and download in background
    HISTORY:
        2017-Nov-16 - Written - Henry Leung (University of Toronto)
    """
    warning_flag = False

    dr = apogee_default_dr(dr=dr)

    if dr == 13:
        file_hash = '5e87eb3ba202f9db24216978dafb19d39d382fc6'

        str1 = 'https://data.sdss.org/sas/dr13/apogee/vac/apogee-rc/cat/'
        filename = 'apogee-rc-DR{}.fits'.format(dr)
        urlstr = str1 + filename
        fullfilename = os.path.join(_APOGEE_DATA, 'dr13/apogee/vac/apogee-rc/cat/')
        if not os.path.exists(fullfilename):
            os.makedirs(fullfilename)
        fullfilename = os.path.join(_APOGEE_DATA, 'dr13/apogee/vac/apogee-rc/cat/', filename)

    elif dr == 14:
        file_hash = '104513070f1c280954f3d1886cac429dbdf2eaf6'

        str1 = 'https://data.sdss.org/sas/dr14/apogee/vac/apogee-rc/cat/'
        filename = 'apogee-rc-DR{}.fits'.format(dr)
        urlstr = str1 + filename
        fullfilename = os.path.join(_APOGEE_DATA, 'dr14/apogee/vac/apogee-rc/cat/')
        if not os.path.exists(fullfilename):
            os.makedirs(fullfilename)
        fullfilename = os.path.join(_APOGEE_DATA, 'dr14/apogee/vac/apogee-rc/cat/', filename)

    else:
        raise ValueError('apogee_vac_rc only supports DR13 or DR14')

    # check file integrity
    if os.path.isfile(fullfilename) and flag is None:
        checksum = sha1_checksum(fullfilename)
        if checksum != file_hash.lower():
            print('File corruption detected, astroNN attempting to download again')
            apogee_vac_rc(dr=dr, verbose=verbose, flag=1)
        else:
            print(fullfilename + ' was found!')

    elif not os.path.isfile(fullfilename) or flag == 1:
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=urlstr.split('/')[-1]) as t:
            urllib.request.urlretrieve(urlstr, fullfilename, reporthook=t.update_to)
            print('Downloaded DR{} Red Clumps Catalog successfully to {}'.format(dr, fullfilename))
            checksum = sha1_checksum(fullfilename)
            if checksum != file_hash.lower():
                print('File corruption detected, astroNN attempting to download again')
                apogee_vac_rc(dr=dr, verbose=verbose, flag=1)

    return fullfilename


def apogee_distances(dr=None, verbose=1, flag=None):
    """
    NAME:
        apogee_distances
    PURPOSE:
        download the red clumps catalogue
    INPUT:
        dr (int): APOGEE DR, example dr=14
    OUTPUT:
        (path): full file path and download in background
    HISTORY:
        2018-Jan-24 - Written - Henry Leung (University of Toronto)
    """
    warning_flag = False

    dr = apogee_default_dr(dr=dr)

    if dr == 14:
        file_hash = 'b33c8419be784b1be3d14af3ee9696c6ac31830f'

        str1 = 'https://data.sdss.org/sas/dr14/apogee/vac/apogee-distances/'
        filename = 'apogee_distances-DR{}.fits'.format(dr)
        urlstr = str1 + filename
        fullfilename = os.path.join(_APOGEE_DATA, 'dr14/apogee/vac/apogee-distances/')
        if not os.path.exists(fullfilename):
            os.makedirs(fullfilename)
        fullfilename = os.path.join(_APOGEE_DATA, 'dr14/apogee/vac/apogee-distances/', filename)

        # check file integrity
        if os.path.isfile(fullfilename) and flag is None:
            checksum = sha1_checksum(fullfilename)
            if checksum != file_hash.lower():
                print('File corruption detected, astroNN attempting to download again')
                apogee_distances(dr=dr, verbose=verbose, flag=1)
            else:
                print(fullfilename + ' was found!')

        elif not os.path.isfile(fullfilename) or flag == 1:
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=urlstr.split('/')[-1]) as t:
                urllib.request.urlretrieve(urlstr, fullfilename, reporthook=t.update_to)
                print('Downloaded DR14 Distances successfully to {}'.format(fullfilename))
                checksum = sha1_checksum(fullfilename)
                if checksum != file_hash.lower():
                    print('File corruption detected, astroNN attempting to download again')
                    apogee_distances(dr=dr, verbose=verbose, flag=1)
    else:
        raise ValueError('apogee_distances only supports DR14')

    return fullfilename
