# ---------------------------------------------------------#
#   astroNN.datasets.galaxy10: galaxy10
# ---------------------------------------------------------#

import os
import urllib.request

import h5py
import numpy as np

from astroNN.config import astroNN_CACHE_DIR
from astroNN.shared.downloader_tools import TqdmUpTo, filehash

Galaxy10Class = {
    0: "Disturbed",
    1: "Merging",
    2: "Round Smooth",
    3: "Smooth, Cigar shaped",
    4: "Cigar Shaped Smooth",
    5: "Barred Spiral",
    6: "Unbarred Tight Spiral",
    7: "Unbarred Loose Spiral",
    8: "Edge-on without Bulge",
    9: "Edge-on with Bulge",
}


_G10_ORIGIN = "https://www.astro.utoronto.ca/~hleung/shared/Galaxy10/"


def load_data(flag=None):
    """
    NAME:
        load_data
    PURPOSE:
        load_data galaxy10 DECals data
    INPUT:
        None
    OUTPUT:
        x (ndarray): An array of images
        y (ndarray): An array of answer
    HISTORY:
        2021-Mar-24 - Written - Henry Leung (University of Toronto)
    """

    filename = "Galaxy10_DECals.h5"

    complete_url = _G10_ORIGIN + filename

    datadir = os.path.join(astroNN_CACHE_DIR, "datasets")
    file_hash = (
        "19AEFC477C41BB7F77FF07599A6B82A038DC042F889A111B0D4D98BB755C1571"  # SHA256
    )

    # Notice python expect sha256 in lowercase

    if not os.path.exists(datadir):
        os.makedirs(datadir)
    fullfilename = os.path.join(datadir, filename)

    # Check if files exists
    if os.path.isfile(fullfilename) and flag is None:
        checksum = filehash(fullfilename, algorithm="sha256")
        if checksum != file_hash.lower():
            print("File corruption detected, astroNN is attempting to download again")
            load_data(flag=1)
        else:
            print(fullfilename + " was found!")
    elif not os.path.isfile(fullfilename) or flag == 1:
        with TqdmUpTo(
            unit="B", unit_scale=True, miniters=1, desc=complete_url.split("/")[-1]
        ) as t:
            urllib.request.urlretrieve(
                complete_url, fullfilename, reporthook=t.update_to
            )
            print(f"Downloaded Galaxy10 successfully to {fullfilename}")
            checksum = filehash(fullfilename, algorithm="sha256")
            if checksum != file_hash.lower():
                load_data(flag=1)

    with h5py.File(fullfilename, "r") as F:
        x = np.array(F["images"])
        y = np.array(F["ans"])

    return x, y


def galaxy10cls_lookup(class_num):
    """
    NAME:
        galaxy10cls_lookup
    PURPOSE:
        look up class name for Galaxy10
    INPUT:
        class_num (int): An integer 0-9
    OUTPUT:
        (string): Name of the class
    HISTORY:
        2018-Feb-07 - Written - Henry Leung (University of Toronto)
    """
    if isinstance(class_num, list) or isinstance(class_num, np.ndarray):
        class_num = np.argmax(class_num)
    if 0 > class_num or 9 < class_num:
        raise ValueError(
            f"Galaxy10 only has 10 classes (class 0 to class 9), you entered class {class_num}"
        )
    return Galaxy10Class[class_num]


def galaxy10_confusion(confusion_mat):
    """
    NAME:
        galaxy10_confusion
    PURPOSE:
        to plot confusion matrix
    INPUT:
        confusion_mat (ndarray): An integer 0-9
    OUTPUT:
        (string): Name of the class
    HISTORY:
        2018-Feb-11 - Written - Henry Leung (University of Toronto)
    """
    import pylab as plt

    conf_arr = confusion_mat.astype(int)

    norm_conf = []
    a = np.max(conf_arr)
    for i in conf_arr:
        tmp_arr = []
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure(figsize=(10, 10.5))
    ax = fig.gca()
    ax.set_title("Confusion Matrix for Galaxy10", fontsize=20)
    ax.set_aspect(1)
    ax.imshow(np.array(norm_conf), cmap=plt.get_cmap("Blues"), interpolation="nearest")

    width, height = conf_arr.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(
                str(conf_arr[x][y]),
                xy=(y, x),
                horizontalalignment="center",
                verticalalignment="center",
            )

    ax.set_xticks(np.arange(width))
    ax.set_xticklabels([str(i) for i in range(width)], fontsize=20)
    ax.set_yticks(np.arange(height))
    ax.set_yticklabels([str(i) for i in range(width)], fontsize=20)
    ax.set_ylabel("Prediction", fontsize=20)
    ax.set_xlabel("Truth", fontsize=20)
    fig.tight_layout()

    return None
