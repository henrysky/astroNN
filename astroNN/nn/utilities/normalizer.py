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

        self.mean_labels = {}
        self.std_labels = {}

        self._custom_norm_func = None
        self._custom_denorm_func = None

    def mode_checker(self, data):
        if type(data) is not dict:
            dict_flag = False
            data = {"Temp": data}
            self.mean_labels = {"Temp": self.mean_labels}
            self.std_labels = {"Temp": self.std_labels}
        else:
            dict_flag = True

        master_data = {}

        for name in data.keys():  # normalize data for each named inputs
            if data[name].ndim == 1:
                data_array = np.expand_dims(data[name], 1)
            else:
                data_array = np.array(data[name])

            self.normalization_mode = str(self.normalization_mode)  # just to prevent unnecessary type issue

            if data_array.dtype == bool:
                if self.normalization_mode != '0':  # binary classification case
                    warnings.warn("Data type is detected as bool, setting normalization_mode to 0 which is "
                                  "doing nothing because no normalization can be done on bool")
                    self.normalization_mode = '0'
                data_array = data_array.astype(np.float)  # need to convert bool to [0., 1.]

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
                self.mean_labels.update({name: np.array([0.])})
                self.std_labels.update({name: np.array([1.])})
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
                self.mean_labels.update({name: np.array([0.])})
                self.std_labels.update({name: np.array([255.])})
            else:
                raise ValueError(f"Unknown Mode -> {self.normalization_mode}")
            master_data.update({name: data_array})

        return master_data, dict_flag

    def normalize(self, data, calc=True):
        data_array, dict_flag = self.mode_checker(data)

        for name in data_array.keys():  # normalize data for each named inputs
            magic_mask = [(data_array[name] == MAGIC_NUMBER)]

            try:
                self.mean_labels[name]
            except KeyError:
                self.mean_labels.update({name: np.array([0.])})
            try:
                self.std_labels[name]
            except KeyError:
                self.std_labels.update({name: np.array([1.])})

            if calc is True:  # check if normalizing with predefine values or get a new one
                print(
                    f"""====Message from {self.__class__.__name__}==== \n You selected mode: {self.normalization_mode} \n Featurewise Center: {self.featurewise_center} \n Datawise Center: {self.datasetwise_center} \n Featurewise std Center: {self.featurewise_stdalization} \n Datawise std Center: {self.datasetwise_stdalization} \n ====Message ends====""")

                if self.featurewise_center is True:
                    self.mean_labels.update({name: np.ma.array(data_array[name], mask=magic_mask).mean(axis=0)})
                    data_array[name] -= self.mean_labels[name]
                elif self.datasetwise_center is True:
                    self.mean_labels.update({name: np.ma.array(data_array[name], mask=magic_mask).mean()})
                    data_array[name] -= self.mean_labels[name]

                if self.featurewise_stdalization is True:
                    self.std_labels.update({name: np.ma.array(data_array[name], mask=magic_mask).std(axis=0)})
                    data_array[name] /= self.std_labels[name]
                elif self.datasetwise_stdalization is True:
                    self.std_labels.update({name: np.ma.array(data_array[name], mask=magic_mask).std()})
                    data_array[name] /= self.std_labels[name]

                if self.normalization_mode == '255':
                    data_array[name] -= self.mean_labels[name]
                    data_array[name] /= self.std_labels[name]
            else:
                data_array[name] -= self.mean_labels[name]
                data_array[name] /= self.std_labels[name]

            if self._custom_norm_func is not None:
                data_array.update({name: self._custom_norm_func(data_array[name])})

            np.place(data_array[name], magic_mask, MAGIC_NUMBER)

        if not dict_flag:
            data_array = data_array['Temp']
            self.mean_labels = self.mean_labels['Temp']
            self.std_labels = self.std_labels['Temp']

        return data_array

    def denormalize(self, data):
        data_array, dict_flag = self.mode_checker(data)
        for name in data_array.keys():  # normalize data for each named inputs
            magic_mask = [data_array[name] == MAGIC_NUMBER]

            if self._custom_denorm_func is not None:
                data_array[name] = self._custom_denorm_func(data_array[name])
            data_array[name] *= self.std_labels[name]
            data_array[name] += self.mean_labels[name]

            np.place(data_array[name], magic_mask, MAGIC_NUMBER)

        if not dict_flag:
            data_array = data_array["Temp"]
            self.mean_labels = self.mean_labels['Temp']
            self.std_labels = self.std_labels['Temp']

        return data_array
