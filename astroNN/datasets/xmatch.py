# ---------------------------------------------------------#
#   astroNN.datasets.xmatch: matching function between catalog
# ---------------------------------------------------------#

import astropy.coordinates as acoords
import numpy as np
from astropy import units as u


# ---------------------------------------------------------#
#   Please notice that astroNN.datasets.xmatch.xmatch() is a modification from Jo Bovy's gaia_tools
#   If you found this xmatch function useful, please take a look at https://github.com/jobovy/gaia_tools/
# ---------------------------------------------------------#


def xmatch(cat1, cat2, maxdist=2, colRA1=1, colDec1=1, epoch1=2000., colRA2=1, colDec2=1, epoch2=2000.,
           colpmRA2=1, colpmDec2=1, swap=False):
    """
    NAME:
       xmatch
    PURPOSE:
       Please notice that astroNN.datasets.xmatch.xmatch() is a modification from Jo Bovy's gaia_tools
       cross-match two catalogs (incl. proper motion in cat2 if epochs are different)
    INPUT:
       cat1 - First catalog
       cat2 - Second catalog
       maxdist= (2) maximum distance in arcsec
       colRA1= ('RA') name of the tag in cat1 with the right ascension in degree in cat1 (assumed to be ICRS)
       colDec1= ('DEC') name of the tag in cat1 with the declination in degree in cat1 (assumed to be ICRS)
       epoch1= (2000.) epoch of the coordinates in cat1
       colRA2= ('RA') name of the tag in cat2 with the right ascension in degree in cat2 (assumed to be ICRS)
       colDec2= ('DEC') name of the tag in cat2 with the declination in degree in cat2 (assumed to be ICRS)
       epoch2= (2000.) epoch of the coordinates in cat2
       colpmRA2= ('pmra') name of the tag in cat2 with the proper motion in right ascension in degree in cat2 (assumed
                          to be ICRS; includes cos(Dec)) [only used when epochs are different]
       colpmDec2= ('pmdec') name of the tag in cat2 with the proper motion in declination in degree in cat2 (assumed to
                            be ICRS) [only used when epochs are different]
       swap= (False) if False, find closest matches in cat2 for each cat1 source, if False do the opposite (important
                      when one of the catalogs has duplicates)
    OUTPUT:
       (index into cat1 of matching objects,
        index into cat2 of matching objects,
        angular separation between matching objects)
    HISTORY:
       2016-09-12 - Written - Bovy (UofT)
       2016-09-21 - Account for Gaia epoch 2015 - Bovy (UofT)
    """

    depoch = epoch2 - epoch1
    if depoch != 0.:
        # Use proper motion to get both catalogs at the same time
        dra = colpmRA2 / np.cos(colDec2 / 180. * np.pi) \
              / 3600000. * depoch
        ddec = colpmDec2 / 3600000. * depoch
    else:
        dra = 0.
        ddec = 0.
    mc1 = acoords.SkyCoord(colRA1, colDec1, unit=(u.degree, u.degree), frame='icrs')
    mc2 = acoords.SkyCoord(colRA2 - dra, colDec2 - ddec, unit=(u.degree, u.degree), frame='icrs')
    if swap:
        idx, d2d, d3d = mc2.match_to_catalog_sky(mc1)
        m1 = np.arange(len(cat2))
    else:
        idx, d2d, d3d = mc1.match_to_catalog_sky(mc2)
        m1 = np.arange(len(cat1))
    mindx = d2d < maxdist * u.arcsec
    m1 = m1[mindx]
    m2 = idx[mindx]
    if swap:
        return (m2, m1, d2d[mindx])
    else:
        return (m1, m2, d2d[mindx])
