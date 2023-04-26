# ---------------------------------------------------------#
#   astroNN.apogee.downloader: download apogee files
# ---------------------------------------------------------#

import getpass
import os
import urllib.request, urllib.error
import warnings

import numpy as np
from astroNN.apogee.apogee_shared import apogee_env, apogee_default_dr
from astroNN.shared.downloader_tools import TqdmUpTo, filehash
from astropy.io import fits
from astroNN.shared import logging as logging

currentdir = os.getcwd()

# global var
warning_flag = False
_ALLSTAR_TEMP = {}
__apogee_credentials_username = None
__apogee_credentials_pw = None


def __apogee_credentials_downloader(url, fullfilename):
    """
    Download file at the URL with apogee credentials, this function will prompt for username and password

    :param url: URL
    :type url: str
    :param fullfilename: Full file name including path in local system
    :type fullfilename: str
    :return: None
    :History: 2018-Aug-31 - Written - Henry Leung (University of Toronto)
    """
    passman = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    global __apogee_credentials_username
    global __apogee_credentials_pw
    if __apogee_credentials_username is None:
        print(
            "\nYou are trying to access APOGEE proprietary data...Please provide username and password..."
        )
        __apogee_credentials_username = input("Username: ")
        __apogee_credentials_pw = getpass.getpass("Password: ")
    passman.add_password(
        None, url, __apogee_credentials_username, __apogee_credentials_pw
    )
    authhandler = urllib.request.HTTPBasicAuthHandler(passman)
    opener = urllib.request.build_opener(authhandler)
    urllib.request.install_opener(opener)
    # Check if directory exists
    if not os.path.exists(os.path.dirname(fullfilename)):
        os.makedirs(os.path.dirname(fullfilename))
    try:
        with TqdmUpTo(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            urllib.request.urlretrieve(url, fullfilename, reporthook=t.update_to)
    except urllib.error.HTTPError as emsg:
        if "401" in str(emsg):
            __apogee_credentials_username = None
            __apogee_credentials_pw = None
            raise ConnectionError("Wrong username or password")
        elif "404" in str(emsg):
            warnings.warn(f"{url} cannot be found on server, skipped")
            fullfilename = warning_flag
        else:
            # don't raise error, so a batch downloading script will keep running despite some files not found
            warnings.warn(f"Unknown error occurred - {emsg}", RuntimeWarning)
            fullfilename = warning_flag

    return fullfilename


def allstar(dr=None, flag=None):
    """
    Download the allStar file (catalog of ASPCAP stellar parameters and abundances from combined spectra)

    :param dr: APOGEE DR
    :type dr: int
    :param flag: 0: normal, 1: force to re-download
    :type flag: int
    :return: full file path and download in background if not found locally, False if cannot be found on server
    :rtype: str
    :History: 2017-Oct-09 - Written - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)

    if dr == 13:
        file_hash = "1718723ada3018de94e1022cd57d4d950a74f91f"

        # Check if directory exists
        fullfoldername = os.path.join(
            apogee_env(), "dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/"
        )
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
        filename = "allStar-l30e.2.fits"
        fullfilename = os.path.join(fullfoldername, filename)
        url = f"https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/{filename}"
    elif dr == 14:
        file_hash = "a7e1801924661954da792e377ad54f412219b105"

        fullfoldername = os.path.join(
            apogee_env(), "dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/"
        )
        # Check if directory exists
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
        filename = "allStar-l31c.2.fits"
        fullfilename = os.path.join(fullfoldername, filename)
        url = f"https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/{filename}"
    elif dr == 16:
        file_hash = "66fe854bd000ca1c0a6b50a998877e4a3e41d184"

        fullfoldername = os.path.join(
            apogee_env(), "dr16/apogee/spectro/aspcap/r12/l33/"
        )
        # Check if directory exists
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
        filename = "allStar-r12-l33.fits"
        fullfilename = os.path.join(fullfoldername, filename)
        url = f"https://data.sdss.org/sas/dr16/apogee/spectro/aspcap/r12/l33/{filename}"
    elif dr == 17:
        file_hash = "7aa2f381de0e8e246f9833cc7da540ef45096702"

        fullfoldername = os.path.join(
            apogee_env(), "dr17/apogee/spectro/aspcap/dr17/synspec_rev1/"
        )
        # Check if directory exists
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
        filename = "allStar-dr17-synspec_rev1.fits"
        fullfilename = os.path.join(fullfoldername, filename)
        url = f"https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/{filename}"
    else:
        raise ValueError("allstar() only supports APOGEE DR13-DR17")

    # check file integrity
    if os.path.isfile(fullfilename) and flag is None:
        checksum = filehash(fullfilename, algorithm="sha1")
        if checksum != file_hash.lower():
            warnings.warn(
                "File corruption detected, astroNN is attempting to download again"
            )
            allstar(dr=dr, flag=1)
        else:
            logging.info(fullfilename + " was found!")

    # Check if files exists
    if not os.path.isfile(os.path.join(fullfoldername, filename)) or flag == 1:
        with TqdmUpTo(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            try:
                urllib.request.urlretrieve(url, fullfilename, reporthook=t.update_to)
                logging.info(
                    f"Downloaded DR{dr:d} allStar file catalog successfully to {fullfilename}"
                )
                checksum = filehash(fullfilename, algorithm="sha1")
                if checksum != file_hash.lower():
                    warnings.warn(
                        "File corruption detected, astroNN is attempting to download again"
                    )
                    allstar(dr=dr, flag=1)
            except urllib.error.HTTPError as emsg:
                if "401" in str(emsg):
                    fullfilename = __apogee_credentials_downloader(url, fullfilename)
                elif "404" in str(emsg):
                    warnings.warn(f"{url} cannot be found on server, skipped")
                    fullfilename = warning_flag
                else:
                    warnings.warn(f"Unknown error occurred - {emsg}")
                    fullfilename = warning_flag

    return fullfilename


def apogee_astronn(dr=None, flag=None):
    """
    Download the apogee_astroNN file (catalog of astroNN stellar parameters, abundances, distances and orbital
     parameters from combined spectra)

    :param dr: APOGEE DR
    :type dr: int
    :param flag: 0: normal, 1: force to re-download
    :type flag: int
    :return: full file path and download in background if not found locally, False if cannot be found on server
    :rtype: str
    :History: 2019-Dec-10 - Written - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)

    if dr == 16:
        # Check if directory exists
        fullfoldername = os.path.join(apogee_env(), "dr16/apogee/vac/apogee-astronn/")
        # Check if directory exists
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
        filename = "apogee_astroNN-DR16-v1.fits"
        fullfilename = os.path.join(fullfoldername, filename)
        file_hash = "1b81ed13eef36fe9a327a05f4a622246522199b2"

        url = f"https://data.sdss.org/sas/dr16/apogee/vac/apogee-astronn/{filename}"
    elif dr == 17:
        # Check if directory exists
        fullfoldername = os.path.join(apogee_env(), "dr17/apogee/vac/apogee-astronn/")
        # Check if directory exists
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
        filename = "apogee_astroNN-DR17.fits"
        fullfilename = os.path.join(fullfoldername, filename)
        file_hash = "c422b9adba840b3415af2fe6dec6500219f1b68f"

        url = f"https://data.sdss.org/sas/dr17/apogee/vac/apogee-astronn/{filename}"
    else:
        raise ValueError("apogee_astroNN() only supports APOGEE DR16-DR17")

    # check file integrity
    if os.path.isfile(fullfilename) and flag is None:
        checksum = filehash(fullfilename, algorithm="sha1")
        if checksum != file_hash.lower():
            warnings.warn(
                "File corruption detected, astroNN is attempting to download again"
            )
            apogee_astronn(dr=dr, flag=1)
        else:
            logging.info(fullfilename + " was found!")

    # Check if files exists
    if not os.path.isfile(os.path.join(fullfoldername, filename)) or flag == 1:
        with TqdmUpTo(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            try:
                urllib.request.urlretrieve(url, fullfilename, reporthook=t.update_to)
                logging.info(
                    f"Downloaded DR{dr:d} apogee_astroNN file catalog successfully to {fullfilename}"
                )
                checksum = filehash(fullfilename, algorithm="sha1")
                if checksum != file_hash.lower():
                    warnings.warn(
                        "File corruption detected, astroNN is attempting to download again"
                    )
                    apogee_astronn(dr=dr, flag=1)
            except urllib.error.HTTPError as emsg:
                if "401" in str(emsg):
                    fullfilename = __apogee_credentials_downloader(url, fullfilename)
                elif "404" in str(emsg):
                    warnings.warn(f"{url} cannot be found on server, skipped")
                    fullfilename = warning_flag
                else:
                    warnings.warn(f"Unknown error occurred - {emsg}")
                    fullfilename = warning_flag

    return fullfilename


def allstar_cannon(dr=None, flag=None):
    """
    Download the allStarCannon file (catalog of Cannon stellar parameters and abundances from combined spectra)

    :param dr: APOGEE DR
    :type dr: int
    :param flag: 0: normal, 1: force to re-download
    :type flag: int
    :return: full file path and download in background if not found locally, False if cannot be found on server
    :rtype: str
    :History: 2017-Oct-24 - Written - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)

    if dr == 14:
        # Check if directory exists
        fullfoldername = os.path.join(
            apogee_env(), "dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/cannon/"
        )
        # Check if directory exists
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
        filename = "allStarCannon-l31c.2.fits"
        fullfilename = os.path.join(fullfoldername, filename)
        file_hash = "64d485e95b3504df0b795ab604e21a71d5c7ae45"

        url = f"https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/cannon/{filename}"
    else:
        raise ValueError("allstar_cannon() only supports APOGEE DR14-DR15")

    # check file integrity
    if os.path.isfile(fullfilename) and flag is None:
        checksum = filehash(fullfilename, algorithm="sha1")
        if checksum != file_hash.lower():
            warnings.warn(
                "File corruption detected, astroNN is attempting to download again"
            )
            allstar_cannon(dr=dr, flag=1)
        else:
            logging.info(fullfilename + " was found!")

    # Check if files exists
    if not os.path.isfile(os.path.join(fullfoldername, filename)) or flag == 1:
        with TqdmUpTo(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            urllib.request.urlretrieve(url, fullfilename, reporthook=t.update_to)
            logging.info(
                f"Downloaded DR{dr:d} allStarCannon file catalog successfully to {fullfilename}"
            )
            checksum = filehash(fullfilename, algorithm="sha1")
            if checksum != file_hash.lower():
                warnings.warn(
                    "File corruption detected, astroNN is attempting to download again"
                )
                allstar_cannon(dr=dr, flag=1)

    return fullfilename


def allvisit(dr=None, flag=None):
    """
    Download the allVisit file (catalog of properties from individual visit spectra)

    :param dr: APOGEE DR
    :type dr: int
    :param flag: 0: normal, 1: force to re-download
    :type flag: int
    :return: full file path and download in background if not found locally, False if cannot be found on server
    :rtype: str
    :History: 2017-Oct-11 - Written - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)

    if dr == 13:
        file_hash = "2a3b13ccd40a2c8aea8321be9630117922d55b51"

        # Check if directory exists
        fullfilepath = os.path.join(apogee_env(), "dr13/apogee/spectro/redux/r6/")
        if not os.path.exists(fullfilepath):
            os.makedirs(fullfilepath)
        filename = "allVisit-l30e.2.fits"
        fullfilename = os.path.join(fullfilepath, filename)
        url = f"https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/{filename}"
    elif dr == 14:
        file_hash = "abcecbcdc5fe8d00779738702c115633811e6bbd"

        # Check if directory exists
        fullfilepath = os.path.join(apogee_env(), "dr14/apogee/spectro/redux/r8/")
        if not os.path.exists(fullfilepath):
            os.makedirs(fullfilepath)
        filename = "allVisit-l31c.2.fits"
        fullfilename = os.path.join(fullfilepath, filename)
        url = f"https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/{filename}"
    elif dr == 16:
        file_hash = "65befb967d8d9d6f4f87711c1fa8d0ac014b62da"

        # Check if directory exists
        fullfilepath = os.path.join(apogee_env(), "dr16/apogee/spectro/aspcap/r12/l33/")
        if not os.path.exists(fullfilepath):
            os.makedirs(fullfilepath)
        filename = "allVisit-r12-l33.fits"
        fullfilename = os.path.join(fullfilepath, filename)
        url = f"https://data.sdss.org/sas/dr16/apogee/spectro/aspcap/r12/l33/{filename}"
    elif dr == 17:
        file_hash = "fb2f5ecbabbe156f8ec37b420e095f3ba8323cc6"

        # Check if directory exists
        fullfilepath = os.path.join(
            apogee_env(), "dr17/apogee/spectro/aspcap/dr17/synspec_rev1/"
        )
        if not os.path.exists(fullfilepath):
            os.makedirs(fullfilepath)
        filename = "allVisit-dr17-synspec_rev1.fits"
        fullfilename = os.path.join(fullfilepath, filename)
        url = f"https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/{filename}"
    else:
        raise ValueError("allvisit() only supports APOGEE DR13-DR17")

    # check file integrity
    if os.path.isfile(fullfilename) and flag is None:
        checksum = filehash(fullfilename, algorithm="sha1")
        if checksum != file_hash.lower():
            warnings.warn(
                "File corruption detected, astroNN is attempting to download again"
            )
            allvisit(dr=dr, flag=1)
        else:
            logging.info(fullfilename + " was found!")
    elif not os.path.isfile(os.path.join(fullfilepath, filename)) or flag == 1:
        with TqdmUpTo(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            urllib.request.urlretrieve(url, fullfilename, reporthook=t.update_to)
            logging.info(
                f"Downloaded DR{dr:d} allVisit file catalog successfully to {fullfilepath}"
            )
            checksum = filehash(fullfilename, algorithm="sha1")
            if checksum != file_hash.lower():
                warnings.warn(
                    "File corruption detected, astroNN is attempting to download again"
                )
                allvisit(dr=dr, flag=1)

    return fullfilename


def combined_spectra(
    dr=None,
    location=None,
    field=None,
    apogee=None,
    telescope=None,
    verbose=1,
    flag=None,
):
    """
    Download the required combined spectra file a.k.a aspcapStar

    :param dr: APOGEE DR
    :type dr: int
    :param location: Location ID [Optional]
    :type location: int
    :param field: Field [Optional]
    :type field: str
    :param apogee: Apogee ID
    :type apogee: str
    :param telescope: Telescope ID, for example 'apo25m' or 'lco25m'
    :type telescope: str
    :param verbose: verbose, set 0 to silent most logging
    :type verbose: int
    :param flag: 0: normal, 1: force to re-download
    :type flag: int

    :return: full file path and download in background if not found locally, False if cannot be found on server
    :rtype: str
    :History:
        | 2017-Oct-15 - Written - Henry Leung (University of Toronto)
        | 2018-Aug-31 - Updated - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)

    # for DR16=<, location is expected to be none because field is used
    if (location is None and dr < 16) or (
        field is None and dr >= 16
    ):  # try to load info if not enough info
        global _ALLSTAR_TEMP
        if not str(f"dr{dr}") in _ALLSTAR_TEMP:
            _ALLSTAR_TEMP[f"dr{dr}"] = fits.getdata(allstar(dr=dr))
        if telescope is None:
            matched_idx = [
                np.nonzero(_ALLSTAR_TEMP[f"dr{dr}"]["APOGEE_ID"] == apogee)[0]
            ][0]
        else:
            matched_idx = [
                np.nonzero(
                    [
                        (_ALLSTAR_TEMP[f"dr{dr}"]["APOGEE_ID"] == apogee)
                        & (_ALLSTAR_TEMP[f"dr{dr}"]["TELESCOPE"] == telescope)
                    ]
                )
            ][0][1]
        if len(matched_idx) == 0:
            raise ValueError(
                f"No entry found in allstar DR{dr} met with your requirement!!"
            )

        location = (
            _ALLSTAR_TEMP[f"dr{dr}"]["LOCATION_ID"][matched_idx][0]
            if not location
            else location
        )
        field = (
            _ALLSTAR_TEMP[f"dr{dr}"]["FIELD"][matched_idx][0] if not field else field
        )
        telescope = (
            _ALLSTAR_TEMP[f"dr{dr}"]["TELESCOPE"][matched_idx][0]
            if not telescope
            else telescope
        )

    if dr == 13:
        reduce_prefix = "r6"
        aspcap_code = "l30e"
        str1 = f"https://data.sdss.org/sas/dr{dr}/apogee/spectro/redux/{reduce_prefix}/stars/{aspcap_code}/{aspcap_code}.2/{location}/"

        filename = f"aspcapStar-{reduce_prefix}-{aspcap_code}.2-{apogee}.fits"
        hash_filename = f"stars_{aspcap_code}_{aspcap_code}.2_{location}.sha1sum"
        urlstr = str1 + filename

        # check folder existence
        fullfoldername = os.path.join(
            apogee_env(),
            f"dr{dr}/apogee/spectro/redux/{reduce_prefix}/stars/{aspcap_code}/{aspcap_code}.2/",
            str(location),
        )
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)

        fullfilename = os.path.join(fullfoldername, filename)

    elif dr == 14:
        reduce_prefix = "r8"
        aspcap_code = "l31c"
        str1 = f"https://data.sdss.org/sas/dr{dr}/apogee/spectro/redux/{reduce_prefix}/stars/{aspcap_code}/{aspcap_code}.2/{location}/"

        filename = f"aspcapStar-{reduce_prefix}-{aspcap_code}.2-{apogee}.fits"
        hash_filename = f"stars_{aspcap_code}_{aspcap_code}.2_{location}.sha1sum"
        urlstr = str1 + filename

        # check folder existence
        fullfoldername = os.path.join(
            apogee_env(),
            f"dr{dr}/apogee/spectro/redux/{reduce_prefix}/stars/{aspcap_code}/{aspcap_code}.2/",
            str(location),
        )
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)

        fullfilename = os.path.join(fullfoldername, filename)
    elif dr == 16:
        reduce_prefix = "r12"
        aspcap_code = "l33"
        str1 = f"https://data.sdss.org/sas/dr16/apogee/spectro/aspcap/{reduce_prefix}/{aspcap_code}/{telescope}/{field}/"

        filename = f"aspcapStar-{reduce_prefix}-{apogee}.fits"
        hash_filename = f"{reduce_prefix}_{aspcap_code}_{telescope}_{field}.sha1sum"
        urlstr = str1 + filename

        # check folder existence
        fullfoldername = os.path.join(
            apogee_env(),
            f"dr{dr}/apogee/spectro/aspcap/{reduce_prefix}/{aspcap_code}/{telescope}",
            str(f"{field}"),
        )
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)

        fullfilename = os.path.join(fullfoldername, filename)
    elif dr == 17:
        reduce_prefix = "dr17"
        aspcap_code = "synspec_rev1"
        str1 = f"https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/{reduce_prefix}/{aspcap_code}/{telescope}/{field}/"

        filename = f"aspcapStar-{reduce_prefix}-{apogee}.fits"
        if telescope == "lco25m":  # syncspec_rev1 only affected lco25m
            hash_filename = f"{reduce_prefix}_{aspcap_code}_{telescope}_{field}.sha1sum"
        else:
            hash_filename = (
                f"{reduce_prefix}_{aspcap_code[:7]}_{telescope}_{field}.sha1sum"
            )
        urlstr = str1 + filename

        # check folder existence
        fullfoldername = os.path.join(
            apogee_env(),
            f"dr{dr}/apogee/spectro/aspcap/{reduce_prefix}/{aspcap_code}/{telescope}",
            str(f"{field}"),
        )
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)

        fullfilename = os.path.join(fullfoldername, filename)
    else:
        raise ValueError("combined_spectra() only supports APOGEE DR13-DR17")

    # check hash file
    full_hash_filename = os.path.join(fullfoldername, hash_filename)
    if not os.path.isfile(full_hash_filename):
        # return warning flag if the location_id cannot even be found
        try:
            urllib.request.urlopen(str1)
        except urllib.error.HTTPError:
            return warning_flag
        urllib.request.urlretrieve(str1 + hash_filename, full_hash_filename)

    hash_list = np.loadtxt(full_hash_filename, dtype="str").T

    # In some rare case, the hash cant be found, so during checking, check len(file_has)!=0 too
    file_hash = hash_list[0][np.argwhere(hash_list[1] == filename)]

    if os.path.isfile(fullfilename) and flag is None:
        checksum = filehash(fullfilename, algorithm="sha1")
        if checksum != file_hash and len(file_hash) != 0:
            warnings.warn(
                "File corruption detected, astroNN is attempting to download again"
            )
            combined_spectra(
                dr=dr, location=location, apogee=apogee, verbose=verbose, flag=1
            )

        if verbose == 1:
            logging.info(fullfilename + " was found!")

    elif not os.path.isfile(fullfilename) or flag == 1:
        try:
            urllib.request.urlretrieve(urlstr, fullfilename)
            logging.info(
                f"Downloaded DR{dr} combined file successfully to {fullfilename}"
            )
            checksum = filehash(fullfilename, algorithm="sha1")
            if checksum != file_hash and len(file_hash) != 0:
                warnings.warn(
                    "File corruption detected, astroNN is attempting to download again"
                )
                combined_spectra(
                    dr=dr, location=location, apogee=apogee, verbose=verbose, flag=1
                )
        except urllib.error.HTTPError as emsg:
            if "401" in str(emsg):
                fullfilename = __apogee_credentials_downloader(urlstr, fullfilename)
            elif "404" in str(emsg):
                warnings.warn(f"{urlstr} cannot be found on server, skipped")
                fullfilename = warning_flag
            else:
                warnings.warn(f"Unknown error occurred - {emsg}")
                fullfilename = warning_flag

    return fullfilename


def visit_spectra(
    dr=None,
    location=None,
    field=None,
    apogee=None,
    telescope=None,
    verbose=1,
    flag=None,
    commission=False,
):
    """
    Download the required individual spectra file a.k.a apStar or asStar

    :param dr: APOGEE DR
    :type dr: int
    :param location: Location ID [Optional]
    :type location: int
    :param field: Field [Optional]
    :type field: str
    :param apogee: Apogee ID
    :type apogee: str
    :param telescope: Telescope ID, for example 'apo25m' or 'lco25m'
    :type telescope: str
    :param verbose: verbose, set 0 to silent most logging
    :type verbose: int
    :param flag: 0: normal, 1: force to re-download
    :type flag: int
    :param commission: whether the spectra is taken during commissioning
    :type commission: bool

    :return: full file path and download in background if not found locally, False if cannot be found on server
    :rtype: str
    :History:
        | 2017-Nov-11 - Written - Henry Leung (University of Toronto)
        | 2018-Aug-31 - Updated - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)

    # for DR16=<, location is expected to be none because field is used
    if (location is None and dr < 16) or (
        field is None and dr >= 16
    ):  # try to load info if not enough info
        global _ALLSTAR_TEMP
        if not str(f"dr{dr}") in _ALLSTAR_TEMP:
            _ALLSTAR_TEMP[f"dr{dr}"] = fits.getdata(allstar(dr=dr))
        if telescope is None:
            matched_idx = [
                np.nonzero(_ALLSTAR_TEMP[f"dr{dr}"]["APOGEE_ID"] == apogee)[0]
            ][0]
        else:
            matched_idx = [
                np.nonzero(
                    [
                        (_ALLSTAR_TEMP[f"dr{dr}"]["APOGEE_ID"] == apogee)
                        & (_ALLSTAR_TEMP[f"dr{dr}"]["TELESCOPE"] == telescope)
                    ]
                )
            ][0][1]
        if len(matched_idx) == 0:
            raise ValueError(
                f"No entry found in allstar DR{dr} met with your requirement!!"
            )

        location = (
            _ALLSTAR_TEMP[f"dr{dr}"]["LOCATION_ID"][matched_idx][0]
            if not location
            else location
        )
        field = (
            _ALLSTAR_TEMP[f"dr{dr}"]["FIELD"][matched_idx][0] if not field else field
        )
        telescope = (
            _ALLSTAR_TEMP[f"dr{dr}"]["TELESCOPE"][matched_idx][0]
            if not telescope
            else telescope
        )

    if dr == 13:
        reduce_prefix = "r6"
        str1 = f"https://data.sdss.org/sas/dr{dr}/apogee/spectro/redux/{reduce_prefix}/stars/apo25m/{location}/"
        if commission:
            filename = f"apStarC-{reduce_prefix}-{apogee}.fits"
        else:
            filename = f"apStar-{reduce_prefix}-{apogee}.fits"
        urlstr = str1 + filename
        hash_filename = f"{reduce_prefix}_stars_apo25m_{location}.sha1sum"

        fullfoldername = os.path.join(
            apogee_env(),
            f"dr{dr}/apogee/spectro/redux/{reduce_prefix}/stars/apo25m/",
            str(location),
        )
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)

    elif dr == 14:
        reduce_prefix = "r8"
        str1 = f"https://data.sdss.org/sas/dr{dr}/apogee/spectro/redux/{reduce_prefix}/stars/apo25m/{location}/"
        if commission:
            filename = f"apStarC-{reduce_prefix}-{apogee}.fits"
        else:
            filename = f"apStar-{reduce_prefix}-{apogee}.fits"
        urlstr = str1 + filename
        hash_filename = f"{reduce_prefix}_stars_apo25m_{location}.sha1sum"

        fullfoldername = os.path.join(
            apogee_env(),
            f"dr{dr}/apogee/spectro/redux/{reduce_prefix}/stars/apo25m/",
            str(location),
        )
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)

    elif dr == 16:
        reduce_prefix = "r12"
        str1 = f"https://data.sdss.org/sas/dr16/apogee/spectro/redux/{reduce_prefix}/stars/{telescope}/{field}/"
        if telescope == "lco25m":
            if commission:
                filename = f"asStarC-{reduce_prefix}-{apogee}.fits"
            else:
                filename = f"asStar-{reduce_prefix}-{apogee}.fits"
        else:
            if commission:
                filename = f"apStarC-{reduce_prefix}-{apogee}.fits"
            else:
                filename = f"apStar-{reduce_prefix}-{apogee}.fits"
        urlstr = str1 + filename
        hash_filename = f"{reduce_prefix}_stars_{telescope}_{field}.sha1sum"

        fullfoldername = os.path.join(
            apogee_env(),
            f"dr{dr}/apogee/spectro/redux/{reduce_prefix}/stars/{telescope}/",
            str(f"{field}"),
        )

        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
    elif dr == 17:
        reduce_prefix = "dr17"
        str1 = f"https://data.sdss.org/sas/dr17/apogee/spectro/redux/{reduce_prefix}/stars/{telescope}/{field}/"
        if telescope == "lco25m":
            if commission:
                filename = f"asStarC-{reduce_prefix}-{apogee}.fits"
            else:
                filename = f"asStar-{reduce_prefix}-{apogee}.fits"
        else:
            if commission:
                filename = f"apStarC-{reduce_prefix}-{apogee}.fits"
            else:
                filename = f"apStar-{reduce_prefix}-{apogee}.fits"
        urlstr = str1 + filename
        hash_filename = f"{reduce_prefix}_stars_{telescope}_{field}.sha1sum"

        fullfoldername = os.path.join(
            apogee_env(),
            f"dr{dr}/apogee/spectro/redux/{reduce_prefix}/stars/{telescope}/",
            str(f"{field}"),
        )

        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
    else:
        raise ValueError("visit_spectra() only supports APOGEE DR13-DR17")

    # check hash file
    full_hash_filename = os.path.join(fullfoldername, hash_filename)
    if not os.path.isfile(full_hash_filename):
        # return warning flag if the location_id cannot even be found
        try:
            urllib.request.urlopen(str1)
        except urllib.error.HTTPError:
            return warning_flag
        urllib.request.urlretrieve(str1 + hash_filename, full_hash_filename)

    hash_list = np.loadtxt(full_hash_filename, dtype="str").T

    fullfilename = os.path.join(fullfoldername, filename)

    # In some rare case, the hash cant be found, so during checking, check len(file_has)!=0 too
    # visit spectra has a different filename in checksum
    # handle the case where apogee_id cannot be found
    hash_idx = [
        i
        for i, item in enumerate(hash_list[1])
        if f"apStar-{reduce_prefix}-{apogee}" in item
    ]
    file_hash = hash_list[0][hash_idx]

    if os.path.isfile(fullfilename) and flag is None:
        checksum = filehash(fullfilename, algorithm="sha1")
        if checksum != file_hash and len(file_hash) != 0:
            warnings.warn(
                "File corruption detected, astroNN is attempting to download again"
            )
            visit_spectra(
                dr=dr, location=location, apogee=apogee, verbose=verbose, flag=1
            )

        if verbose:
            logging.info(fullfilename + " was found!")

    elif not os.path.isfile(fullfilename) or flag == 1:
        try:
            urllib.request.urlretrieve(urlstr, fullfilename)
            logging.info(
                f"Downloaded DR{dr} individual visit file successfully to {fullfilename}"
            )
            checksum = filehash(fullfilename, algorithm="sha1")
            if checksum != file_hash and len(file_hash) != 0:
                warnings.warn(
                    "File corruption detected, astroNN is attempting to download again"
                )
                visit_spectra(
                    dr=dr, location=location, apogee=apogee, verbose=verbose, flag=1
                )
        except urllib.error.HTTPError as emsg:
            if "401" in str(emsg):
                fullfilename = __apogee_credentials_downloader(urlstr, fullfilename)
            elif "404" in str(emsg):
                warnings.warn(f"{urlstr} cannot be found on server, skipped")
                fullfilename = warning_flag
            else:
                warnings.warn(f"Unknown error occurred - {emsg}")
                fullfilename = warning_flag

    return fullfilename


def apogee_rc(dr=None, flag=None):
    """
    Download the APOGEE red clumps catalogue

    :param dr: Apogee DR
    :type dr: int
    :param flag: Force to download if flag=1
    :type flag: int
    :return: full file path
    :rtype: str
    :History: 2017-Nov-16 - Written - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)

    if dr == 13:
        file_hash = "5e87eb3ba202f9db24216978dafb19d39d382fc6"

        str1 = "https://data.sdss.org/sas/dr13/apogee/vac/apogee-rc/cat/"
        filename = f"apogee-rc-DR{dr}.fits"
        urlstr = str1 + filename
        fullfoldername = os.path.join(apogee_env(), "dr13/apogee/vac/apogee-rc/cat/")
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
        fullfilename = os.path.join(fullfoldername, filename)

    elif dr == 14:
        file_hash = "104513070f1c280954f3d1886cac429dbdf2eaf6"

        str1 = "https://data.sdss.org/sas/dr14/apogee/vac/apogee-rc/cat/"
        filename = f"apogee-rc-DR{dr}.fits"
        urlstr = str1 + filename
        fullfoldername = os.path.join(apogee_env(), "dr14/apogee/vac/apogee-rc/cat/")
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
        fullfilename = os.path.join(fullfoldername, filename)

    elif dr == 16:
        file_hash = "0bc75a230058f50ed8a5ea3fa8554d803ffc103d"

        str1 = "https://data.sdss.org/sas/dr16/apogee/vac/apogee-rc/cat/"
        filename = f"apogee-rc-DR{dr}.fits"
        urlstr = str1 + filename
        fullfoldername = os.path.join(apogee_env(), "dr16/apogee/vac/apogee-rc/cat/")
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
        fullfilename = os.path.join(fullfoldername, filename)

    elif dr == 17:
        file_hash = "491e854d6db6b828554eda2b4b2e31365ccf65aa"

        str1 = "https://data.sdss.org/sas/dr17/apogee/vac/apogee-rc/cat/"
        filename = f"apogee-rc-DR{dr}.fits"
        urlstr = str1 + filename
        fullfoldername = os.path.join(apogee_env(), "dr17/apogee/vac/apogee-rc/cat/")
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
        fullfilename = os.path.join(fullfoldername, filename)

    else:
        raise ValueError("apogee_rc() only supports APOGEE DR13-DR17")

    # check file integrity
    if os.path.isfile(fullfilename) and flag is None:
        checksum = filehash(fullfilename, algorithm="sha1")
        if checksum != file_hash.lower():
            warnings.warn(
                "File corruption detected, astroNN is attempting to download again"
            )
            apogee_rc(dr=dr, flag=1)
        else:
            logging.info(fullfilename + " was found!")

    elif not os.path.isfile(fullfilename) or flag == 1:
        try:
            with TqdmUpTo(
                unit="B", unit_scale=True, miniters=1, desc=urlstr.split("/")[-1]
            ) as t:
                urllib.request.urlretrieve(urlstr, fullfilename, reporthook=t.update_to)
                logging.info(
                    f"Downloaded DR{dr} Red Clumps Catalog successfully to {fullfilename}"
                )
                checksum = filehash(fullfilename, algorithm="sha1")
                if checksum != file_hash.lower():
                    warnings.warn(
                        "File corruption detected, astroNN is attempting to download again"
                    )
                    apogee_rc(dr=dr, flag=1)
        except urllib.error.HTTPError as emsg:
            if "401" in str(emsg):
                fullfilename = __apogee_credentials_downloader(urlstr, fullfilename)
            elif "404" in str(emsg):
                warnings.warn(f"{urlstr} cannot be found on server, skipped")
                fullfilename = warning_flag
            else:
                warnings.warn(f"Unknown error occurred - {emsg}")
                fullfilename = warning_flag

    return fullfilename


def apogee_distances(dr=None, flag=None):
    """
    Download the APOGEE Distances VAC catalogue (APOGEE Distances for DR14, APOGEE Starhourse for DR16/17)

    :param dr: APOGEE DR
    :type dr: int
    :param flag: Force to download if flag=1
    :type flag: int
    :return: full file path
    :rtype: str
    :History:
        | 2018-Jan-24 - Written - Henry Leung (University of Toronto)
        | 2021-Jan-29 - Updated - Henry Leung (University of Toronto)
    """
    dr = apogee_default_dr(dr=dr)

    if dr == 14:
        file_hash = "b33c8419be784b1be3d14af3ee9696c6ac31830f"

        str1 = "https://data.sdss.org/sas/dr14/apogee/vac/apogee-distances/"
        filename = f"apogee_distances-DR{dr}.fits"
        urlstr = str1 + filename
        fullfoldername = os.path.join(apogee_env(), "dr14/apogee/vac/apogee-distances/")
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
        fullfilename = os.path.join(fullfoldername, filename)
    if dr == 16:
        file_hash = "2502e2f7703046163f81ecc4054dce39b2038e4f"

        str1 = "https://data.sdss.org/sas/dr16/apogee/vac/apogee-starhorse/"
        filename = f"apogee_starhorse-DR{dr}-v1.fits"
        urlstr = str1 + filename
        fullfoldername = os.path.join(apogee_env(), "dr16/apogee/vac/apogee-starhorse/")
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
        fullfilename = os.path.join(fullfoldername, filename)
    if dr == 17:
        file_hash = "2502e2f7703046163f81ecc4054dce39b2038e4f"

        str1 = "https://data.sdss.org/sas/dr17/apogee/vac/apogee-starhorse/"
        filename = f"APOGEE_DR17_EDR3_STARHORSE_v2.fits"
        urlstr = str1 + filename
        fullfoldername = os.path.join(apogee_env(), "dr17/apogee/vac/apogee-starhorse/")
        if not os.path.exists(fullfoldername):
            os.makedirs(fullfoldername)
        fullfilename = os.path.join(fullfoldername, filename)
    else:
        raise ValueError("apogee_distances() only supports APOGEE DR14-DR17")

    # check file integrity
    if os.path.isfile(fullfilename) and flag is None:
        checksum = filehash(fullfilename, algorithm="sha1")
        if checksum != file_hash.lower():
            warnings.warn(
                "File corruption detected, astroNN is attempting to download again"
            )
            apogee_distances(dr=dr, flag=1)
        else:
            logging.info(fullfilename + " was found!")

    elif not os.path.isfile(fullfilename) or flag == 1:
        try:
            with TqdmUpTo(
                unit="B", unit_scale=True, miniters=1, desc=urlstr.split("/")[-1]
            ) as t:
                urllib.request.urlretrieve(urlstr, fullfilename, reporthook=t.update_to)
                logging.info(
                    f"Downloaded DR{dr} Distances successfully to {fullfilename}"
                )
                checksum = filehash(fullfilename, algorithm="sha1")
                if checksum != file_hash.lower():
                    warnings.warn(
                        "File corruption detected, astroNN is attempting to download again"
                    )
                    apogee_distances(dr=dr, flag=1)
        except urllib.error.HTTPError:
            warnings.warn(f"{urlstr} cannot be found on server, skipped")
            fullfilename = warning_flag

    return fullfilename
