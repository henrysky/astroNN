###############################################################################
#   normalizer.py: top-level class for normalizer
###############################################################################
import warnings
import numpy as np

from astroNN.config import MAGIC_NUMBER
from astroNN.nn.numpy import sigmoid_inv, sigmoid
from astroNN.shared.dict_tools import list_to_dict, to_iterable


class Normalizer(object):
    """Top-level class for a normalizer"""

    def __init__(self, mode=None, verbose=2):
        """
        To define a normalizer

        :param mode: normalization mode
        :type mode: int
        :param verbose: level of verbose
        :type verbose: int

        :History: 2018-Jan-06 - Written - Henry Leung (University of Toronto)
        """

        self.normalization_mode = mode
        self.verbose = verbose

        self.featurewise_center = {}
        self.datasetwise_center = {}

        self.featurewise_stdalization = {}
        self.datasetwise_stdalization = {}

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
        if type(self.normalization_mode) is not dict:
            self.normalization_mode = list_to_dict(
                data.keys(), to_iterable(self.normalization_mode)
            )
        for name in data.keys():  # normalize data for each named inputs
            if data[name].ndim == 1:
                data_array = np.expand_dims(data[name], 1)
            else:
                data_array = np.array(data[name])

            self.normalization_mode.update(
                {name: str(self.normalization_mode[name])}
            )  # just to prevent unnecessary type issue

            if data_array.dtype == bool:
                if self.normalization_mode[name] != "0":  # binary classification case
                    warnings.warn(
                        "Data type is detected as bool, setting normalization_mode to 0 which is "
                        "doing nothing because no normalization can be done on bool"
                    )
                    self.normalization_mode[name] = "0"
            data_array = data_array.astype(
                np.float32, copy=False
            )  # need to convert data to float in every case

            if self.normalization_mode[name] == "0":
                self.featurewise_center.update({name: False})
                self.datasetwise_center.update({name: False})
                self.featurewise_stdalization.update({name: False})
                self.datasetwise_stdalization.update({name: False})
                self.mean_labels.update({name: np.array([0.0])})
                self.std_labels.update({name: np.array([1.0])})
            elif self.normalization_mode[name] == "1":
                self.featurewise_center.update({name: False})
                self.datasetwise_center.update({name: True})
                self.featurewise_stdalization.update({name: False})
                self.datasetwise_stdalization.update({name: True})
            elif self.normalization_mode[name] == "2":
                self.featurewise_center.update({name: True})
                self.datasetwise_center.update({name: False})
                self.featurewise_stdalization.update({name: True})
                self.datasetwise_stdalization.update({name: False})
            elif self.normalization_mode[name] == "3":
                self.featurewise_center.update({name: True})
                self.datasetwise_center.update({name: False})
                self.featurewise_stdalization.update({name: False})
                self.datasetwise_stdalization.update({name: False})
            elif (
                self.normalization_mode[name] == "3s"
            ):  # allow custom function, default to use sigmoid to normalize
                self.featurewise_center.update({name: False})
                self.datasetwise_center.update({name: False})
                self.featurewise_stdalization.update({name: False})
                self.datasetwise_stdalization.update({name: False})
                if self._custom_norm_func is None:
                    self._custom_norm_func = sigmoid
                if self._custom_denorm_func is None:
                    self._custom_denorm_func = sigmoid_inv
                self.mean_labels.update({name: np.array([0.0])})
                self.std_labels.update({name: np.array([1.0])})
            elif self.normalization_mode[name] == "4":
                self.featurewise_center.update({name: False})
                self.datasetwise_center.update({name: False})
                self.featurewise_stdalization.update({name: True})
                self.datasetwise_stdalization.update({name: False})
            elif self.normalization_mode[name] == "255":
                # Used to normalize 8bit images
                self.featurewise_center.update({name: False})
                self.datasetwise_center.update({name: False})
                self.featurewise_stdalization.update({name: False})
                self.datasetwise_stdalization.update({name: False})
                self.mean_labels.update({name: np.array([0.0])})
                self.std_labels.update({name: np.array([255.0])})
            else:
                raise ValueError(f"Unknown Mode -> {self.normalization_mode[name]}")
            master_data.update({name: data_array})

        return master_data, dict_flag

    def normalize(self, data, calc=True):
        data_array, dict_flag = self.mode_checker(data)

        for name in data_array.keys():  # normalize data for each named inputs
            magic_mask = [
                (data_array[name] == MAGIC_NUMBER) | (np.isnan(data_array[name]))
            ]

            try:
                self.mean_labels[name]
            except KeyError:
                self.mean_labels.update({name: np.array([0.0])})
            try:
                self.std_labels[name]
            except KeyError:
                self.std_labels.update({name: np.array([1.0])})

            if (
                calc is True
            ):  # check if normalizing with predefine values or get a new one
                if self.verbose > 0:
                    print(
                        f"""====Message from {self.__class__.__name__}====
You selected mode: {self.normalization_mode[name]}
Featurewise Center: {self.featurewise_center}
Datawise Center: {self.datasetwise_center} 
Featurewise std Center: {self.featurewise_stdalization}
Datawise std Center: {self.datasetwise_stdalization} 
====Message ends===="""
                    )

                if self.featurewise_center[name] is True:
                    self.mean_labels.update(
                        {
                            name: np.ma.array(data_array[name], mask=magic_mask).mean(
                                axis=0
                            )
                        }
                    )
                    data_array[name] -= self.mean_labels[name]
                elif self.datasetwise_center[name] is True:
                    self.mean_labels.update(
                        {name: np.ma.array(data_array[name], mask=magic_mask).mean()}
                    )
                    data_array[name] -= self.mean_labels[name]

                if self.featurewise_stdalization[name] is True:
                    self.std_labels.update(
                        {
                            name: np.ma.array(data_array[name], mask=magic_mask).std(
                                axis=0
                            )
                        }
                    )
                    data_array[name] /= self.std_labels[name]
                elif self.datasetwise_stdalization[name] is True:
                    self.std_labels.update(
                        {name: np.ma.array(data_array[name], mask=magic_mask).std()}
                    )
                    data_array[name] /= self.std_labels[name]
                if self.normalization_mode[name] == "255":
                    data_array[name] -= self.mean_labels[name]
                    data_array[name] /= self.std_labels[name]
            else:
                data_array[name] -= self.mean_labels[name]
                data_array[name] /= self.std_labels[name]

            if self._custom_norm_func is not None:
                data_array.update({name: self._custom_norm_func(data_array[name])})

            np.place(data_array[name], magic_mask, MAGIC_NUMBER)

        if not dict_flag:
            data_array = data_array["Temp"]
            self.mean_labels = self.mean_labels["Temp"]
            self.std_labels = self.std_labels["Temp"]

        return data_array

    def denormalize(self, data):
        data_array, dict_flag = self.mode_checker(data)
        for name in data_array.keys():  # normalize data for each named inputs
            magic_mask = [
                (data_array[name] == MAGIC_NUMBER) | (np.isnan(data_array[name]))
            ]

            if self._custom_denorm_func is not None:
                data_array[name] = self._custom_denorm_func(data_array[name])
            data_array[name] *= self.std_labels[name]
            data_array[name] += self.mean_labels[name]

            np.place(data_array[name], magic_mask, MAGIC_NUMBER)

        if not dict_flag:
            data_array = data_array["Temp"]
            self.mean_labels = self.mean_labels["Temp"]
            self.std_labels = self.std_labels["Temp"]

        return data_array
