# ---------------------------------------------------------#
#   astroNN.apogee.downloader: download apogee files
# ---------------------------------------------------------#

import os
import urllib.request

import numpy as np

from astroNN.apogee.apogee_shared import apogee_env, apogee_default_dr
from astroNN.shared.downloader_tools import TqdmUpTo
from astroNN.shared.downloader_tools import sha1_checksum

currentdir = os.getcwd()

_APOGEE_DATA = apogee_env()
warning_flag = False


def allstar(dr=None, flag=None):
    """
    NAME:
        allstar
    PURPOSE:
        download the allStar file (catalog of ASPCAP stellar parameters and abundances from combined spectra)
    INPUT:
        dr (int): APOGEE DR, example dr=14
        flag (int): 0: normal, 1: force to re-download
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
        url = f'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/{filename}'
    elif dr == 14:
        file_hash = 'a7e1801924661954da792e377ad54f412219b105'

        fullfoldername = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/')
        # Check if directory exists
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
        filename = 'allStar-l31c.2.fits'
        fullfilename = os.path.join(fullfoldername, filename)
        url = f'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/{filename}'
    else:
        raise ValueError('allstar() only supports APOGEE DR13 and DR14')

    # check file integrity
    if os.path.isfile(fullfilename) and flag is None:
        checksum = sha1_checksum(fullfilename)
        if checksum != file_hash.lower():
            print('File corruption detected, astroNN attempting to download again')
            allstar(dr=dr, flag=1)
        else:
            print(fullfilename + ' was found!')

    # Check if files exists
    if not os.path.isfile(os.path.join(fullfoldername, filename)) or flag == 1:
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, fullfilename, reporthook=t.update_to)
            print(f'Downloaded DR{dr:d} allStar file catalog successfully to {fullfilename}')
            checksum = sha1_checksum(fullfilename)
            if checksum != file_hash.lower():
                print('File corruption detected, astroNN attempting to download again')
                allstar(dr=dr, flag=1)

    return fullfilename


def allstarcannon(dr=None, flag=None):
    """
    NAME:
        allstarcanon
    PURPOSE:
        download the allStarCannon file (catalog of Cannon stellar parameters and abundances from combined spectra)
    INPUT:
        dr (int): APOGEE DR, example dr=14
        flag (int): 0: normal, 1: force to re-download
    OUTPUT:
        (path): full file path and download in background
    HISTORY:
        2017-Oct-24 - Written - Henry Leung (University of Toronto)
    """

    dr = apogee_default_dr(dr=dr)

    if dr == 14:
        # Check if directory exists
        fullfoldername = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/cannon/')
        # Check if directory exists
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
        filename = 'allStarCannon-l31c.2.fits'
        fullfilename = os.path.join(fullfoldername, filename)
        file_hash = '64d485e95b3504df0b795ab604e21a71d5c7ae45'

        url = f'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/cannon/{filename}'
    elif dr == 13:
        raise ValueError('allstarcanon() currently not supporting DR13')
    else:
        raise ValueError('allstarcannon() only supports APOGEE DR14')

    # check file integrity
    if os.path.isfile(fullfilename) and flag is None:
        checksum = sha1_checksum(fullfilename)
        if checksum != file_hash.lower():
            print('File corruption detected, astroNN attempting to download again')
            allstarcannon(dr=dr, flag=1)
        else:
            print(fullfilename + ' was found!')

    # Check if files exists
    if not os.path.isfile(os.path.join(fullfoldername, filename)) or flag == 1:
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, fullfilename, reporthook=t.update_to)
            print(f'Downloaded DR{dr:d} allStarCannon file catalog successfully to {fullfilename}')
            checksum = sha1_checksum(fullfilename)
            if checksum != file_hash.lower():
                print('File corruption detected, astroNN attempting to download again')
                allstarcannon(dr=dr, flag=1)

    return fullfilename


def allvisit(dr=None, flag=None):
    """
    NAME:
        allvisit
    PURPOSE:
        download the allVisit file (catalog of properties from individual visit spectra)
    INPUT:
        dr (int): APOGEE DR, example dr=14
        flag (int): 0: normal, 1: force to re-download
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
        url = f'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/{filename}'
    elif dr == 14:
        file_hash = 'abcecbcdc5fe8d00779738702c115633811e6bbd'

        # Check if directory exists
        fullfilepath = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/')
        if not os.path.exists(fullfilepath):
            os.makedirs(fullfilepath)
        filename = 'allVisit-l31c.2.fits'
        fullfilename = os.path.join(fullfilepath, filename)
        url = f'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/{filename}'
    else:
        raise ValueError('allvisit() only supports APOGEE DR13 and DR14')

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
            print(f'Downloaded DR{dr:d} allVisit file catalog successfully to {fullfilepath}')
            checksum = sha1_checksum(fullfilename)
            if checksum != file_hash.lower():
                print('File corruption detected, astroNN attempting to download again')
                allstar(dr=dr, flag=1)

    return fullfilename


def combined_spectra(dr=None, location=None, apogee=None, verbose=1, flag=None):
    """
    NAME:
        combined_spectra
    PURPOSE:
        download the required combined spectra file (catalog of properties from individual visit spectra)
    INPUT:
        dr (int): APOGEE DR, example dr=14
        flag (int): 0: normal, 1: force to re-download
    OUTPUT:
        (path): full file path and download in background
    HISTORY:
        2017-Oct-15 - Written - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)

    if dr == 13:
        str1 = f'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/{location}/'

        filename = f'aspcapStar-r6-l30e.2-{apogee}.fits'
        hash_filename = f'stars_l30e_l30e.2_{location}.sha1sum'
        urlstr = str1 + filename

        # check folder existence
        fullfoldername = os.path.join(_APOGEE_DATA, 'dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/', str(location))
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)

        fullfilename = os.path.join(_APOGEE_DATA, 'dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/', str(location),
                                    filename)

    elif dr == 14:
        str1 = f'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/{location}/'

        filename = f'aspcapStar-r8-l31c.2-{apogee}.fits'
        hash_filename = f'stars_l31c_l31c.2_{location}.sha1sum'
        urlstr = str1 + filename

        # check folder existence
        fullfoldername = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/', str(location))
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)

        fullfilename = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/', str(location),
                                    filename)
    else:
        raise ValueError('combined_spectra() only supports DR13 or DR14')

    # check hash file
    full_hash_filename = os.path.join(fullfoldername, hash_filename)
    if not os.path.isfile(full_hash_filename):
        # return warning flag if the location_id cannot even be found
        try:
            urllib.request.urlopen(str1)
        except urllib.request.HTTPError:
            return warning_flag
        urllib.request.urlretrieve(str1 + hash_filename, full_hash_filename)

    hash_list = np.loadtxt(full_hash_filename, dtype='str').T

    # In some rare case, the hash cant be found, so during checking, check len(file_has)!=0 too
    file_hash = (hash_list[0])[np.argwhere(hash_list[1] == filename)]

    if os.path.isfile(fullfilename) and flag is None:
        checksum = sha1_checksum(fullfilename)
        if checksum != file_hash and len(file_hash) != 0:
            print('File corruption detected, astroNN attempting to download again')
            combined_spectra(dr=dr, location=location, apogee=apogee, verbose=verbose, flag=1)

        if verbose == 1:
            print(fullfilename + ' was found!')

    elif not os.path.isfile(fullfilename) or flag == 1:
        try:
            urllib.request.urlretrieve(urlstr, fullfilename)
            print(f'Downloaded DR14 combined file successfully to {fullfilename}')
            checksum = sha1_checksum(fullfilename)
            if checksum != file_hash and len(file_hash) != 0:
                print('File corruption detected, astroNN attempting to download again')
                combined_spectra(dr=dr, location=location, apogee=apogee, verbose=verbose, flag=1)
        except urllib.request.HTTPError:
            print(f'{urlstr} cannot be found on server, skipped')
            fullfilename = warning_flag

    return fullfilename


def visit_spectra(dr=None, location=None, apogee=None, verbose=1, flag=None):
    """
    NAME:
        visit_spectra
    PURPOSE:
        download the combined spectra file (catalog of properties from individual visit spectra)
    INPUT:
        dr (int): APOGEE DR, example dr=14
        flag (int): 0: normal, 1: force to re-download
    OUTPUT:
        (path): full file path and download in background
    HISTORY:
        2017-Nov-11 - Written - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)

    if dr == 13:
        str1 = f'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/apo25m/{location}/'

        filename = f'apStar-r6-{apogee}.fits'
        urlstr = str1 + filename
        hash_filename = f'r6_stars_apo25m_{location}.sha1sum'

        fullfoldername = os.path.join(_APOGEE_DATA, 'dr13/apogee/spectro/redux/r6/stars/apo25m/', str(location))
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)

        # check hash file
        full_hash_filename = os.path.join(fullfoldername, hash_filename)
        if not os.path.isfile(full_hash_filename):
            # return warning flag if the location_id cannot even be found
            try:
                urllib.request.urlopen(str1)
            except urllib.request.HTTPError:
                return warning_flag
            urllib.request.urlretrieve(str1 + hash_filename, full_hash_filename)

        hash_list = np.loadtxt(full_hash_filename, dtype='str').T

        # In some rare case, the hash cant be found, so during checking, check len(file_has)!=0 too
        # visit spectra has a different filename in checksum
        hash_idx = [i for i,item in enumerate(hash_list[1]) if f'apStar-r6-{apogee}' in item][0]
        file_hash = hash_list[0][hash_idx]

        fullfilename = os.path.join(_APOGEE_DATA, 'dr13/apogee/spectro/redux/r6/stars/apo25m/', str(location), filename)

    elif dr == 14:
        str1 = f'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/apo25m/{location}/'

        filename = f'apStar-r8-{apogee}.fits'
        urlstr = str1 + filename
        hash_filename = f'r8_stars_apo25m_{location}.sha1sum'

        fullfoldername = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/apo25m/', str(location))
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)

        # check hash file
        full_hash_filename = os.path.join(fullfoldername, hash_filename)
        if not os.path.isfile(full_hash_filename):
            # return warning flag if the location_id cannot even be found
            try:
                urllib.request.urlopen(str1)
            except urllib.request.HTTPError:
                return warning_flag

            urllib.request.urlretrieve(str1 + hash_filename, full_hash_filename)

        hash_list = np.loadtxt(full_hash_filename, dtype='str').T

        # In some rare case, the hash cant be found, so during checking, check len(file_has)!=0 too
        # visit spectra has a different filename in checksum
        hash_idx = [i for i,item in enumerate(hash_list[1]) if f'apStar-r8-{apogee}' in item][0]
        file_hash = hash_list[0][hash_idx]

        fullfilename = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/apo25m/', str(location), filename)

    else:
        raise ValueError('visit_spectra() only supports DR13 or DR14')

    if os.path.isfile(fullfilename) and flag is None:
        checksum = sha1_checksum(fullfilename)
        if checksum != file_hash and len(file_hash) != 0:
            print('File corruption detected, astroNN attempting to download again')
            visit_spectra(dr=dr, location=location, apogee=apogee, verbose=verbose, flag=1)

        if verbose == 1:
            print(fullfilename + ' was found!')

    elif not os.path.isfile(fullfilename) or flag == 1:
        try:
            urllib.request.urlretrieve(urlstr, fullfilename)
            print(f'Downloaded DR14 individual visit file successfully to {fullfilename}')
            checksum = sha1_checksum(fullfilename)
            if checksum != file_hash and len(file_hash) != 0:
                print('File corruption detected, astroNN attempting to download again')
                visit_spectra(dr=dr, location=location, apogee=apogee, verbose=verbose, flag=1)
        except urllib.request.HTTPError:
            print(f'{urlstr} cannot be found on server, skipped')
            fullfilename = warning_flag
    return fullfilename


def apogee_vac_rc(dr=None, flag=None):
    """
    NAME:
        apogee_vac_rc
    PURPOSE:
        download the red clumps catalogue
    INPUT:
        dr (int): APOGEE DR, example dr=14
        flag (int): 0: normal, 1: force to re-download
    OUTPUT:
        (path): full file path and download in background
    HISTORY:
        2017-Nov-16 - Written - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)

    if dr == 13:
        file_hash = '5e87eb3ba202f9db24216978dafb19d39d382fc6'

        str1 = 'https://data.sdss.org/sas/dr13/apogee/vac/apogee-rc/cat/'
        filename = f'apogee-rc-DR{dr}.fits'
        urlstr = str1 + filename
        fullfilename = os.path.join(_APOGEE_DATA, 'dr13/apogee/vac/apogee-rc/cat/')
        if not os.path.exists(fullfilename):
            os.makedirs(fullfilename)
        fullfilename = os.path.join(_APOGEE_DATA, 'dr13/apogee/vac/apogee-rc/cat/', filename)

    elif dr == 14:
        file_hash = '104513070f1c280954f3d1886cac429dbdf2eaf6'

        str1 = 'https://data.sdss.org/sas/dr14/apogee/vac/apogee-rc/cat/'
        filename = f'apogee-rc-DR{dr}.fits'
        urlstr = str1 + filename
        fullfilename = os.path.join(_APOGEE_DATA, 'dr14/apogee/vac/apogee-rc/cat/')
        if not os.path.exists(fullfilename):
            os.makedirs(fullfilename)
        fullfilename = os.path.join(_APOGEE_DATA, 'dr14/apogee/vac/apogee-rc/cat/', filename)

    else:
        raise ValueError('apogee_vac_rc() only supports DR13 or DR14')

    # check file integrity
    if os.path.isfile(fullfilename) and flag is None:
        checksum = sha1_checksum(fullfilename)
        if checksum != file_hash.lower():
            print('File corruption detected, astroNN attempting to download again')
            apogee_vac_rc(dr=dr, flag=1)
        else:
            print(fullfilename + ' was found!')

    elif not os.path.isfile(fullfilename) or flag == 1:
        try:
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=urlstr.split('/')[-1]) as t:
                urllib.request.urlretrieve(urlstr, fullfilename, reporthook=t.update_to)
                print(f'Downloaded DR{dr} Red Clumps Catalog successfully to {fullfilename}')
                checksum = sha1_checksum(fullfilename)
                if checksum != file_hash.lower():
                    print('File corruption detected, astroNN attempting to download again')
                    apogee_vac_rc(dr=dr, flag=1)
        except urllib.request.HTTPError:
            print(f'{urlstr} cannot be found on server, skipped')
            fullfilename = warning_flag

    return fullfilename


def apogee_distances(dr=None, flag=None):
    """
    NAME:
        apogee_distances
    PURPOSE:
        download the red clumps catalogue
    INPUT:
        dr (int): APOGEE DR, example dr=14
        flag (int): 0: normal, 1: force to re-download
    OUTPUT:
        (path): full file path and download in background
    HISTORY:
        2018-Jan-24 - Written - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)

    if dr == 14:
        file_hash = 'b33c8419be784b1be3d14af3ee9696c6ac31830f'

        str1 = 'https://data.sdss.org/sas/dr14/apogee/vac/apogee-distances/'
        filename = f'apogee_distances-DR{dr}.fits'
        urlstr = str1 + filename
        fullfilename = os.path.join(_APOGEE_DATA, 'dr14/apogee/vac/apogee-distances/')
        if not os.path.exists(fullfilename):
            os.makedirs(fullfilename)
        fullfilename = os.path.join(_APOGEE_DATA, 'dr14/apogee/vac/apogee-distances/', filename)
    else:
        raise ValueError('apogee_distances() only supports DR14')

    # check file integrity
    if os.path.isfile(fullfilename) and flag is None:
        checksum = sha1_checksum(fullfilename)
        if checksum != file_hash.lower():
            print('File corruption detected, astroNN attempting to download again')
            apogee_distances(dr=dr, flag=1)
        else:
            print(fullfilename + ' was found!')

    elif not os.path.isfile(fullfilename) or flag == 1:
        try:
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=urlstr.split('/')[-1]) as t:
                urllib.request.urlretrieve(urlstr, fullfilename, reporthook=t.update_to)
                print(f'Downloaded DR14 Distances successfully to {fullfilename}')
                checksum = sha1_checksum(fullfilename)
                if checksum != file_hash.lower():
                    print('File corruption detected, astroNN attempting to download again')
                    apogee_distances(dr=dr, flag=1)
        except urllib.request.HTTPError:
            print(f'{urlstr} cannot be found on server, skipped')
            fullfilename = warning_flag

    return fullfilename
