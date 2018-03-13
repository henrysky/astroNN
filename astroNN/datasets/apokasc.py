# ---------------------------------------------------------#
#   astroNN.datasets.apokasc: apokasc Log(g)
# ---------------------------------------------------------#

import numpy as np
from astroquery.vizier import Vizier


def apokasc_load(combine=True):
    """
    NAME:
        apokasc_load
    PURPOSE:
        load apokasc result (Precise surface gravity measurement)
    INPUT:
        combine (boolean): True to combine gold snd basic standard
    OUTPUT:
    HISTORY:
        2017-Dec-23 - Written - Henry Leung (University of Toronto)
    """
    catalog_list = Vizier.find_catalogs('apokasc')
    Vizier.ROW_LIMIT = 99999
    catalogs_gold = Vizier.get_catalogs(catalog_list.keys())[1]
    catalogs_basic = Vizier.get_catalogs(catalog_list.keys())[2]
    gold_ra = catalogs_gold['_RA']
    gold_dec = catalogs_gold['_DE']
    gold_logg = catalogs_gold['log_g_']
    basic_ra = catalogs_basic['_RA']
    basic_dec = catalogs_basic['_DE']
    basic_logg = catalogs_basic['log.g2']

    if combine is True:
        ra = np.append(np.array(gold_ra), np.array(basic_ra))
        dec = np.append(np.array(gold_dec), np.array(basic_dec))
        logg = np.append(np.array(gold_logg), np.array(basic_logg))
        return ra, dec, logg
    else:
        return gold_ra, gold_dec, gold_logg, basic_ra, basic_dec, basic_logg
