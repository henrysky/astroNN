###############################################################################
#   normalizer.py: top-level class for normalizer
###############################################################################
import numpy as np


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

        self.featurewise_std_normalization = False
        self.datasetwise_std_normalization = False

    def normalize(self, data):

        if self.normalization_mode == 0:
            self.featurewise_center = False
            self.datasetwise_center = False
            self.featurewise_std_normalization = False
            self.datasetwise_std_normalization = False
        elif self.normalization_mode == 1:
            self.featurewise_center = False
            self.datasetwise_center = True
            self.featurewise_std_normalization = False
            self.datasetwise_std_normalization = True
        elif self.normalization_mode == 2:
            self.featurewise_center = True
            self.datasetwise_center = False
            self.featurewise_std_normalization = True
            self.datasetwise_std_normalization = False

        print('====Message from {}===='.format(self.__class__.__name__))
        print('You selected mode: {}'.format(self.normalization_mode))
        print('Featurewise Center: {}'.format(self.featurewise_center))
        print('Datawise Center: {}'.format(self.datasetwise_center))
        print('Featurewise std Center: {}'.format(self.featurewise_std_normalization))
        print('Datawise std Center: {}'.format(self.datasetwise_std_normalization))
        print('====Message ends====')

        mean_labels = 0
        std_labels = 1
        data_array = np.array(data)

        magic_number = -9999.

        if self.featurewise_center is True:
            mean_labels = np.zeros(data_array.shape[1])
            for i in range(data_array.shape[1]):
                mean_labels[i] = np.mean(data_array[(data_array != magic_number)], axis=0)
                data_array[:, i] -= mean_labels[i]
                (data_array[:, i])[(data_array[:, i] == (magic_number - mean_labels))] = magic_number

        if self.datasetwise_center is True:
            mean_labels = np.mean(data_array[(data_array != magic_number)])
            data_array -= mean_labels
            data_array[(data_array == (magic_number - mean_labels))] = magic_number

        if self.featurewise_std_normalization is True:
            std_labels = np.ones(data_array.shape[1])
            for i in range(data_array.shape[1]):
                std_labels[i] = np.std(data_array[(data_array != magic_number)], axis=0)
                data_array[:, i] /= std_labels[i]
                (data_array[:, i])[(data_array[:, i] == (magic_number / std_labels))] = magic_number
        if self.datasetwise_center is True:
            std_labels = np.std(data_array[(data_array != magic_number)])
            data_array /= std_labels
            data_array[(data_array == (magic_number / std_labels))] = magic_number

        return data_array, mean_labels, std_labels


class Denormalizer(object):
    """Top-level class for a neural network"""

    def __init__(self, mode=1):
        """
        NAME:
            __init__
        PURPOSE:
            To define a denormalizer
        HISTORY:
            2018-Jan-06 - Written - Henry Leung (University of Toronto)
        """

        self.normalization_mode = mode

        self.featurewise_center = False
        self.samplewise_center = False

        self.featurewise_std_normalization = False
        self.samplewise_std_normalization = False

    def denormalize(self, data):
        pass
