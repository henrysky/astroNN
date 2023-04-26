# ---------------------------------------------------------#
#   astroNN.datasets.h5: compile h5 files for NN
# ---------------------------------------------------------#

import os
from functools import reduce

import h5py
import numpy as np


def h5name_check(h5name):
    if h5name is None:
        raise ValueError('Please specify the dataset name using filename="..."')
    return None


class H5Loader(object):
    def __init__(self, filename, target="all"):
        self.filename = filename
        self.target = target
        self.currentdir = os.getcwd()
        self.load_combined = True
        self.load_err = False
        self.exclude9999 = False

        if os.path.isfile(os.path.join(self.currentdir, self.filename)) is True:
            self.h5path = os.path.join(self.currentdir, self.filename)
        elif (
            os.path.isfile(os.path.join(self.currentdir, (self.filename + ".h5")))
            is True
        ):
            self.h5path = os.path.join(self.currentdir, (self.filename + ".h5"))
        else:
            raise FileNotFoundError(
                f"Cannot find {os.path.join(self.currentdir, self.filename)}"
            )

    def load_allowed_index(self):
        with h5py.File(self.h5path) as F:  # ensure the file will be cleaned up
            if self.exclude9999 is True:
                index_not9999 = None
                for counter, tg in enumerate(self.target):
                    if index_not9999 is None:
                        index_not9999 = np.arange(F[f"{tg}"].shape[0])
                    temp_index = np.where(np.array(F[f"{tg}"]) != -9999)[0]
                    index_not9999 = reduce(np.intersect1d, (index_not9999, temp_index))

                in_flag = index_not9999
                if self.load_combined is True:
                    in_flag = np.where(np.array(F["in_flag"]) == 0)[0]
                elif self.load_combined is False:
                    in_flag = np.where(np.array(F["in_flag"]) == 1)[0]

                allowed_index = reduce(np.intersect1d, (index_not9999, in_flag))

            else:
                in_flag = []
                if self.load_combined is True:
                    in_flag = np.where(np.array(F["in_flag"]) == 0)[0]
                elif self.load_combined is False:
                    in_flag = np.where(np.array(F["in_flag"]) == 1)[0]

                allowed_index = in_flag

            F.close()

        return allowed_index

    def load(self):
        allowed_index = self.load_allowed_index()
        with h5py.File(self.h5path) as F:  # ensure the file will be cleaned up
            allowed_index_list = allowed_index.tolist()
            spectra = np.array(F["spectra"])[allowed_index_list]
            spectra_err = np.array(F["spectra_err"])[allowed_index_list]

            y = np.array((spectra.shape[1]))
            y_err = np.array((spectra.shape[1]))
            for counter, tg in enumerate(self.target):
                temp = np.array(F[f"{tg}"])[allowed_index_list]
                if counter == 0:
                    y = temp[:]
                else:
                    y = np.column_stack((y, temp[:]))
                if self.load_err is True:
                    temp_err = np.array(F[f"{tg}_err"])[allowed_index_list]
                    if counter == 0:
                        y_err = temp_err[:]
                    else:
                        y_err = np.column_stack((y_err, temp_err[:]))

        if self.load_err is True:
            return spectra, y, spectra_err, y_err
        else:
            return spectra, y

    def load_entry(self, name):
        """
        NAME:
            load_entry
        PURPOSE:
            load extra entry for the h5loader, the order will be the same as the output from load()
        INPUT:
            name (string): dataset name to laod
        OUTPUT:
            (ndarray): the dataset
        HISTORY:
            2018-Feb-08 - Written - Henry Leung (University of Toronto)
        """
        allowed_index = self.load_allowed_index()
        allowed_index_list = allowed_index.tolist()
        with h5py.File(self.h5path) as F:  # ensure the file will be cleaned up
            return np.array(F[f"{name}"])[allowed_index_list]
