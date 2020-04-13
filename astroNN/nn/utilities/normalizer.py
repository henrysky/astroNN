###############################################################################
#   normalizer.py: top-level class for normalizer
###############################################################################
import warnings
import numpy as np

from astroNN.config import MAGIC_NUMBER
from astroNN.nn.numpy import sigmoid_inv, sigmoid


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

        self._custom_norm_func = None
        self._custom_denorm_func = None

    def mode_checker(self, data):

        if data.ndim == 1:
            data_array = np.expand_dims(data, 1)
        else:
            data_array = np.array(data)

        self.normalization_mode = str(self.normalization_mode)  # just to prevent unnecessary type issue

        if data_array.dtype == bool:
            if self.normalization_mode != '0':
                warnings.warn("Data type is detected as bool, setting normalization_mode to 0 which is doing nothing "
                              "because no normalization can be done on bool")
                self.normalization_mode = '0'
            data_array = data_array.astype(np.float)

        if self.normalization_mode == '0':
            self.featurewise_center = False
            self.datasetwise_center = False
            self.featurewise_stdalization = False
            self.datasetwise_stdalization = False
        elif self.normalization_mode == '1':
            self.featurewise_center = False
            self.datasetwise_center = True
            self.featurewise_stdalization = False
            self.datasetwise_stdalization = True
        elif self.normalization_mode == '2':
            self.featurewise_center = True
            self.datasetwise_center = False
            self.featurewise_stdalization = True
            self.datasetwise_stdalization = False
        elif self.normalization_mode == '3':
            self.featurewise_center = True
            self.datasetwise_center = False
            self.featurewise_stdalization = False
            self.datasetwise_stdalization = False
        elif self.normalization_mode == '3s':  # allow custom function, default to use sigmoid to normalize
            self.featurewise_center = True
            self.datasetwise_center = False
            self.featurewise_stdalization = False
            self.datasetwise_stdalization = False
            if self._custom_norm_func is None:
                self._custom_norm_func = sigmoid
            if self._custom_denorm_func is None:
                self._custom_denorm_func = sigmoid_inv
        elif self.normalization_mode == '4':
            self.featurewise_center = False
            self.datasetwise_center = False
            self.featurewise_stdalization = True
            self.datasetwise_stdalization = False
        elif self.normalization_mode == '255':
            # Used to normalize 8bit images
            self.featurewise_center = False
            self.datasetwise_center = False
            self.featurewise_stdalization = False
            self.datasetwise_stdalization = False
            self.mean_labels = np.array([0.])
            self.std_labels = np.array([255.])
        else:
            raise ValueError(f"Unknown Mode -> {self.normalization_mode}")

        return data_array

    def normalize(self, data, calc=True):
        data_array = self.mode_checker(data)

        magic_mask = [(data_array == MAGIC_NUMBER)]

        if calc is True:  # check if normalizing with predefine values or get a new one
            print(f'====Message from {self.__class__.__name__}====')
            print(f'You selected mode: {self.normalization_mode}')
            print(f'Featurewise Center: {self.featurewise_center}')
            print(f'Datawise Center: {self.datasetwise_center}')
            print(f'Featurewise std Center: {self.featurewise_stdalization}')
            print(f'Datawise std Center: {self.datasetwise_stdalization}')
            print('====Message ends====')

            if self.featurewise_center is True:
                self.mean_labels = np.ma.array(data_array, mask=magic_mask).mean(axis=0)
                data_array -= self.mean_labels
            elif self.datasetwise_center is True:
                self.mean_labels = np.ma.array(data_array, mask=magic_mask).mean()
                data_array -= self.mean_labels

            if self.featurewise_stdalization is True:
                self.std_labels = np.ma.array(data_array, mask=magic_mask).std(axis=0)
                data_array /= self.std_labels
            elif self.datasetwise_stdalization is True:
                self.std_labels = np.ma.array(data_array, mask=magic_mask).std()
                data_array /= self.std_labels

            if self.normalization_mode == '255':
                data_array -= self.mean_labels
                data_array /= self.std_labels
        else:
            data_array -= self.mean_labels
            data_array /= self.std_labels

        if self._custom_norm_func is not None:
            data_array = self._custom_norm_func(data_array)

        np.place(data_array, magic_mask, MAGIC_NUMBER)

        return data_array

    def denormalize(self, data):
        data_array = self.mode_checker(data)

        magic_mask = [data_array == MAGIC_NUMBER]

        if self._custom_denorm_func is not None:
            data_array = self._custom_denorm_func(data_array)

        data_array *= self.std_labels
        data_array += self.mean_labels

        np.place(data_array, magic_mask, MAGIC_NUMBER)

        return data_array
