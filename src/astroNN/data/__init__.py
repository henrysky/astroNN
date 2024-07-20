# ---------------------------------------------------------#
#   astroNN.data.__init__: tools for loading data
# ---------------------------------------------------------#
import os

import astroNN


def datapath():
    """
    Get astroNN embedded data path

    :return: full path to embedded data folder
    :rtype: str
    :History:
    | 2018-May-16 - Written - Henry Leung (University of Toronto)
    | 2019-July-02 - Updated - Henry Leung (University of Toronto)
    """
    return os.path.join(os.path.dirname(astroNN.__path__[0]), "astroNN", "data")


def data_description():
    """
    Print data description for astroNN embedded data

    :History:
    | 2018-May-16 - Written - Henry Leung (University of Toronto)
    | 2019-July-02 - Updated - Henry Leung (University of Toronto)
    """
    items = {
        "aspcap_l31c_masks.npy": "ASPCAP l31c (DR14) elements windows mask represented by bits on the 7514px spectrum",
        "dr13_contmask.npz": "APOGEE DR13 continuum mask from Bovy's APOGEE tools",
        "dr14_contmask.npz": "APOGEE DR14 continuum mask from Bovy's APOGEE tools",
        "dr16_contmask.npz": "APOGEE DR16 continuum mask",
        "dr17_contmask.npz": "APOGEE DR17 continuum mask",
    }

    for item in items:
        print(item, ": ", items[item])
