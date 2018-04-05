###############################################################################
#   normalizer.py: top-level class for normalizer
###############################################################################
import numpy as np

from astroNN.config import MAGIC_NUMBER


class Normalizer(object):
    """Top-level class for a normalizer"""

    def __init__(self, mode=None):
        """
        NAME:
            __init__
        PURPOSE:
            To define a normalizer
        HISTORY:
            2018-Jan-06 - Written - Henry Leung (University of Toronto)
        """

        self.normalization_mode = mode

        self.featurewise_center = False
        self.datasetwise_center = False

        self.featurewise_stdalization = False
        self.datasetwise_stdalization = False

        self.mean_labels = np.array([0.])
        self.std_labels = np.array([1.])

    def mode_checker(self):
        if self.normalization_mode == 0:
            self.featurewise_center = False
            self.datasetwise_center = False
            self.featurewise_stdalization = False
            self.datasetwise_stdalization = False
        elif self.normalization_mode == 1:
            self.featurewise_center = False
            self.datasetwise_center = True
            self.featurewise_stdalization = False
            self.datasetwise_stdalization = True
        elif self.normalization_mode == 2:
            self.featurewise_center = True
            self.datasetwise_center = False
            self.featurewise_stdalization = True
            self.datasetwise_stdalization = False
        elif self.normalization_mode == 3:
            self.featurewise_center = True
            self.datasetwise_center = False
            self.featurewise_stdalization = False
            self.datasetwise_stdalization = False
        elif self.normalization_mode == 255:
            # Used to normalize 8bit images
            self.featurewise_center = False
            self.datasetwise_center = False
            self.featurewise_stdalization = False
            self.datasetwise_stdalization = False
            self.mean_labels = np.array([127.5])
            self.std_labels = np.array([127.5])

    def normalize(self, data):
        self.mode_checker()

        print(f'====Message from {self.__class__.__name__}====')
        print(f'You selected mode: {self.normalization_mode}')
        print(f'Featurewise Center: {self.featurewise_center}')
        print(f'Datawise Center: {self.datasetwise_center}')
        print(f'Featurewise std Center: {self.featurewise_stdalization}')
        print(f'Datawise std Center: {self.datasetwise_stdalization}')
        print('====Message ends====')

        if data.ndim == 1:
            data_array = np.expand_dims(data, 1)
        else:
            data_array = np.array(data)

        if self.featurewise_center is True:
            self.mean_labels = np.zeros(data_array.shape[1])
            for i in range(data_array.shape[1]):
                not9999_index = np.where(data_array[:, i] != MAGIC_NUMBER)
                self.mean_labels[i] = np.mean((data_array[:, i])[not9999_index], axis=0)
                (data_array[:, i])[not9999_index] -= self.mean_labels[i]

        if self.datasetwise_center is True:
            self.mean_labels = np.mean(data_array[(data_array != MAGIC_NUMBER)])
            data_array[(data_array != MAGIC_NUMBER)] -= self.mean_labels

        if self.featurewise_stdalization is True:
            self.std_labels = np.ones(data_array.shape[1])
            for i in range(data_array.shape[1]):
                not9999_index = np.where(data_array[:, i] != MAGIC_NUMBER)
                self.std_labels[i] = np.std((data_array[:, i])[not9999_index], axis=0)
                (data_array[:, i])[not9999_index] /= self.std_labels[i]

        if self.datasetwise_stdalization is True:
            self.std_labels = np.std(data_array[(data_array != MAGIC_NUMBER)])
            data_array[(data_array != MAGIC_NUMBER)] /= self.std_labels

        if self.normalization_mode == 255:
            data_array -= self.mean_labels
            data_array /= self.std_labels

        return data_array

    def denormalize(self, data):
        self.mode_checker()

        if data.ndim == 1:
            data_array = np.expand_dims(data, 1)
        else:
            data_array = np.array(data)

        if self.featurewise_stdalization is True:
            for i in range(data_array.shape[1]):
                not9999_index = np.where(data_array[:, i] != MAGIC_NUMBER)
                (data_array[:, i])[not9999_index] *= self.std_labels

        if self.datasetwise_center is True:
            data_array[(data_array != MAGIC_NUMBER)] *= self.std_labels

        if self.featurewise_center is True:
            for i in range(data_array.shape[1]):
                not9999_index = np.where(data_array[:, i] != MAGIC_NUMBER)
                (data_array[:, i])[not9999_index] += self.mean_labels

        if self.datasetwise_stdalization is True:
            data_array[(data_array != MAGIC_NUMBER)] += self.mean_labels

        if self.normalization_mode == 255:
            data_array *= self.std_labels
            data_array += self.mean_labels

        return data_array
