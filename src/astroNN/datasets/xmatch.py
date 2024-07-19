# ---------------------------------------------------------#
#   astroNN.datasets.xmatch: matching function between catalog
# ---------------------------------------------------------#

import pathlib
import numpy as np
import h5py
from astropy.io import fits
from astropy import units as u
import astropy.coordinates as acoords


def xmatch(
    ra1,
    dec1,
    ra2,
    dec2,
    epoch1=2000.0,
    epoch2=2000.0,
    pmra2=None,
    pmdec2=None,
    maxdist=2,
):
    """
    Cross-matching between arrays by RA/DEC coordiantes

    :param ra1: 1d array for the first catalog RA
    :type ra1: ndarray
    :param dec1: 1d array for the first catalog DEC
    :type dec1: ndarray
    :param ra2: 1d array for the second catalog RA
    :type ra2: ndarray
    :param dec2: 1d array for the second catalog DEC
    :type dec2: ndarray
    :param epoch1: Epoch for the first catalog, can be float or 1d array
    :type epoch1: Union([float, ndarray])
    :param epoch1: Epoch for the second catalog, can be float or 1d array
    :type epoch1: Union([float, ndarray])
    :param pmra2: RA proper motion for second catalog, only effective if `epoch1` not equals `epoch2`
    :type pmra2: ndarray
    :param pmdec2: DEC proper motion for second catalog, only effective if `epoch1` not equals `epoch2`
    :type pmdec2: ndarray
    :param maxdist: Maximium distance in arcsecond
    :type maxdist: float

    :return: numpy array of ra, dec, separation
    :rtype: ndarrays
    :History:
        | 2018-Jan-25 - Written - Henry Leung (University of Toronto)
        | 2021-Jan-29 - Updated - Henry Leung (University of Toronto)
    """
    depoch = epoch2 - epoch1
    if np.any(depoch != 0.0):
        # Use proper motion to get both catalogs at the same time
        dra = pmra2 / np.cos(dec2 / 180.0 * np.pi) / 3600000.0 * depoch
        ddec = pmdec2 / 3600000.0 * depoch
    else:
        dra = 0.0
        ddec = 0.0
    mc1 = acoords.SkyCoord(ra1, dec1, unit=(u.degree, u.degree), frame="icrs")
    mc2 = acoords.SkyCoord(
        ra2 - dra, dec2 - ddec, unit=(u.degree, u.degree), frame="icrs"
    )

    idx, d2d, d3d = mc1.match_to_catalog_sky(mc2)
    # to make sure filtering out all neg ones which are untouched
    mindx = (d2d < maxdist * u.arcsec) & (0.0 * u.arcsec <= d2d)
    m1 = np.arange(len(ra1))[mindx]
    m2 = idx[mindx]

    return m1, m2, d2d[mindx]


def xmatch_cat(
    cat1=None,
    cat2=None,
    maxdist=2.0,
    ra1="ra",
    dec1="dec",
    epoch1=2000.0,
    ra2="ra",
    dec2="dec",
    epoch2=2000.0,
    pmra2="pmra",
    pmdec2="pmdec",
    field=None,
):
    """
    Cross-matching between two catalog files by RA/DEC coordiantes

    :param cat1: Catalog 1, can be path to `.fits` or `.h5` or opened fits or h5 files
    :type cat1: str
    :param cat2: Catalog 2, can be path to `.fits` or `.h5` or opened fits or h5 files
    :type cat2: str
    :param maxdist: Maximium distance in arcsecond
    :type maxdist: float
    :param ra1: Field for RA in Catalog 1
    :type ra1: str
    :param dec1: Field for DEC in Catalog 1
    :type dec1: str
    :param ra2: Field for RA in Catalog 2
    :type ra2: str
    :param dec2: Field for DEC in Catalog 2
    :type dec2: str
    :type maxdist: str
    :param maxdist: Maximium distance in arcsecond
    :type maxdist: float
    :param field: Additional field name, if not None then cross-match objects within the same field between the two catalog
    :type field: str

    :return: numpy array of ra, dec, array, err_array
    :rtype: ndarrays
    :History:
        | 2021-Jan-29 - Written - Henry Leung (University of Toronto)
    """
    if isinstance(cat1, (str, pathlib.Path)):
        cat1_ext = pathlib.Path(cat1).suffix
        if cat1_ext.lower == ".h5":
            cat1 = h5py.File(cat1, mode="r")
        elif cat1_ext.lower == ".fits" or cat1_ext.lower == ".fit":
            cat1 = fits.getdata(cat1)
        else:
            raise TypeError(f"Unsupported file type {cat1_ext}")

    if isinstance(cat2, (str, pathlib.Path)):
        cat2_ext = pathlib.Path(cat2).suffix
        if cat2_ext.lower == ".h5":
            cat2 = h5py.File(cat2, mode="r")
        elif cat2_ext.lower == ".fits" or cat2_ext.lower == ".fit":
            cat2 = fits.getdata(cat2)
        else:
            raise TypeError(f"Unsupported file type {cat2_ext}")

    depoch = epoch2 - epoch1

    if field is not None:
        try:  # check if the field actually exists in both cat1/cat2
            cat1[field]
            cat2[field]
        except KeyError:  # python 2/3 format string
            raise KeyError(f"'{field}' does not exist in both catalog")

        uniques = np.unique(cat1[field])

        d2d = np.ones(len(cat1)) * -1.0
        idx = np.zeros(len(cat1), dtype=int)

        for unique in uniques:  # loop over the class
            idx_1 = np.arange(cat1[ra1].shape[0])[cat1[field] == unique]
            idx_2 = np.arange(cat2[ra2].shape[0])[cat2[field] == unique]
            idx1, idx2, sep = xmatch(
                cat1[ra1],
                cat1[dec1],
                cat2[ra2],
                cat2[dec2],
                epoch1=epoch1,
                epoch2=epoch2,
                pmra2=None if np.all(depoch == 0.0) else cat2[pmra2],
                pmdec2=None if np.all(depoch == 0.0) else cat2[pmdec2],
                maxdist=maxdist,
            )

    else:
        idx1, idx2, sep = xmatch(
            cat1[ra1],
            cat1[dec1],
            cat2[ra2],
            cat2[dec2],
            epoch1=epoch1,
            epoch2=epoch2,
            pmra2=None if np.all(depoch == 0.0) else cat2[pmra2],
            pmdec2=None if np.all(depoch == 0.0) else cat2[pmdec2],
            maxdist=maxdist,
        )

    return idx1, idx2, sep
