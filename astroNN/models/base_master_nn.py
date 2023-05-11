###############################################################################
#   base_master_nn.py: top-level class for a neural network
###############################################################################
import os
import sys
import time
import warnings
import pathlib
from abc import ABC, abstractmethod

import numpy as np
import pylab as plt
import keras
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.python.keras.utils.layer_utils import count_params

import astroNN
from astroNN.config import _astroNN_MODEL_NAME
from astroNN.config import cpu_gpu_check
from astroNN.shared.nn_tools import folder_runnum

epsilon, plot_model = tfk.backend.epsilon, tfk.utils.plot_model


class NeuralNetMaster(ABC):
    """
    Top-level class for an astroNN neural network

    :ivar name: Full English name
    :ivar _model_type: Type of model
    :ivar _model_identifier: Unique model identifier, by default using class name as ID
    :ivar _implementation_version: Version of the model
    :ivar _python_info: Placeholder to store python version used for debugging purpose
    :ivar _astronn_ver: astroNN version detected
    :ivar _keras_ver: Keras version detected
    :ivar _tf_ver: Tensorflow version detected
    :ivar currentdir: Current directory of the terminal
    :ivar folder_name: Folder name to be saved
    :ivar fullfilepath: Full file path
    :ivar batch_size: Batch size for training, by default 64
    :ivar autosave: Boolean to flag whether autosave model or not

    :ivar task: Task
    :ivar lr: Learning rate
    :ivar max_epochs: Maximum epochs
    :ivar val_size: Validation set size in percentage
    :ivar val_num: Validation set autual number

    :ivar beta_1: Exponential decay rate for the 1st moment estimates for optimization algorithm
    :ivar beta_2: Exponential decay rate for the 2nd moment estimates for optimization algorithm
    :ivar optimizer_epsilon: A small constant for numerical stability for optimization algorithm
    :ivar optimizer: Placeholder for optimizer

    :ivar targetname: Full name for every output neurones

    :History:
        | 2017-Dec-23 - Written - Henry Leung (University of Toronto)
        | 2018-Jan-05 - Updated - Henry Leung (University of Toronto)
    """

    def __init__(self):
        self.name = None
        self._model_type = None
        self._model_identifier = self.__class__.__name__  # No effect, will do when save
        self._implementation_version = None
        self._python_info = sys.version
        self._astronn_ver = astroNN.__version__
        self._keras_ver = keras.__version__
        self._tf_ver = tf.__version__
        self.currentdir = os.getcwd()
        self.folder_name = None
        self.fullfilepath = None
        self.batch_size = 64
        self.autosave = False

        # Hyperparameter
        self.task = None
        self.lr = None
        self.max_epochs = None
        self.val_size = None
        self.has_val = False  # flag if doing validation or not, if val_size > 0 then means doing validation
        self.val_num = None

        # optimizer parameter
        self.beta_1 = 0.9  # exponential decay rate for the 1st moment estimates for optimization algorithm
        self.beta_2 = 0.999  # exponential decay rate for the 2nd moment estimates for optimization algorithm
        self.optimizer_epsilon = (
            epsilon()
        )  # a small constant for numerical stability for optimization algorithm
        self.optimizer = None

        # Keras API
        self.verbose = 2
        self.keras_model = None
        self.keras_model_predict = None
        self.history = None
        self.metrics = None
        self.callbacks = None
        self.__callbacks = None  # for internal default callbacks usage only
        self._output_loss = None

        self.input_normalizer = None
        self.labels_normalizer = None
        self.training_generator = None
        self.validation_generator = None

        self.input_norm_mode = None
        self.labels_norm_mode = None
        self.input_mean = None
        self.input_std = None
        self.labels_mean = None
        self.labels_std = None
        self.input_names = None
        self.output_names = None

        self._input_shape = None
        self._labels_shape = None

        self.num_train = None
        self.train_idx = None
        self.val_idx = None
        self.targetname = None
        self.history = None
        self.virtual_cvslogger = None
        self.hyper_txt = None

        cpu_gpu_check()

    def __str__(self):
        return f"Name: {self.name}\nModel Type: {self._model_type}\nModel ID: {self._model_identifier}"

    @property
    def has_model(self):
        """
        Get whether the instance has a model, usually a model is created after you called train(), the instance
        will has no model if you did not call train()

        :return: bool
        :History: 2018-May-21 - Written - Henry Leung (University of Toronto)
        """
        if self.keras_model is None:
            return False
        else:
            return True

    def has_model_check(self):
        if self.has_model is False:
            raise AttributeError(
                "No model found in this instance, the common problem is you did not train a model"
            )

    def custom_train_step(self, *args):
        raise NotImplementedError

    def custom_test_step(self, *args):
        raise NotImplementedError

    @abstractmethod
    def train(self, *args):
        raise NotImplementedError

    @abstractmethod
    def train_on_batch(self, *args):
        raise NotImplementedError

    @abstractmethod
    def test(self, *args):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, *args):
        raise NotImplementedError

    @abstractmethod
    def model(self):
        raise NotImplementedError

    @abstractmethod
    def post_training_checklist_child(self):
        raise NotImplementedError

    def _tensor_dict_sanitize(self, tensor_dict, names_list):
        """
        Remove extra tensors

        :param tensor_dict: Dictionary of array or tensors
        :type tensor_dict: dict
        :param names_list: List of names
        :type names_list: list
        :return: Sanitized dict
        """
        for tensor_name in [n for n in tensor_dict.keys() if n not in names_list]:
            tensor_dict.pop(tensor_name)

        return tensor_dict

    def pre_training_checklist_master(self, input_data, labels):
        # handle named inputs/outputs first
        try:
            self.input_names = list(input_data.keys())
        except AttributeError:
            self.input_names = ["input"]  # default input name in all astroNN models
            input_data = {"input": input_data}
        try:
            self.output_names = list(labels.keys())
        except AttributeError:
            self.output_names = ["output"]  # default input name in all astroNN models
            labels = {"output": labels}

        # assert all named input has the same number of data points
        # TODO: add detail error msg, add test
        if not all(
            input_data["input"].shape[0] == input_data[name].shape[0]
            for name in self.input_names
        ):
            raise IndexError("all inputs should contain same number of data point")
        if not all(
            labels["output"].shape[0] == labels[name].shape[0]
            for name in self.output_names
        ):
            raise IndexError("all outputs should contain same number of data point")

        if self.val_size is None:
            self.val_size = 0

        self.val_num = int(input_data["input"].shape[0] * self.val_size)
        self.num_train = input_data["input"].shape[0] - self.val_num
        self.has_val = self.val_num > 0

        # Assuming the convolutional layer immediately after input layer
        # only require if it is new, no need for fine-tuning
        # in case you read this for dense network, use Flattener as first layer in your network to flatten it
        if self._input_shape is None:
            self._input_shape = {}
            for name in self.input_names:
                data_ndim = input_data[name].ndim
                if data_ndim == 1:
                    self._input_shape.update(
                        {
                            name: (
                                1,
                                1,
                            )
                        }
                    )
                elif data_ndim == 2:
                    self._input_shape.update(
                        {
                            name: (
                                input_data[name].shape[1],
                                1,
                            )
                        }
                    )
                elif data_ndim == 3:
                    self._input_shape.update(
                        {
                            name: (
                                input_data[name].shape[1],
                                input_data[name].shape[2],
                                1,
                            )
                        }
                    )
                elif data_ndim == 4:
                    self._input_shape.update(
                        {
                            name: (
                                input_data[name].shape[1],
                                input_data[name].shape[2],
                                input_data[name].shape[3],
                            )
                        }
                    )

            # zeroth dim should always be number of data
            self._labels_shape = {}
            for name in self.output_names:
                data_ndim = labels[name].ndim
                if data_ndim == 1:
                    self._labels_shape.update({name: 1})
                elif data_ndim == 2:
                    self._labels_shape.update({name: (labels[name].shape[1])})
                elif data_ndim == 3:
                    self._labels_shape.update(
                        {name: (labels[name].shape[1], labels[name].shape[2])}
                    )
                elif data_ndim == 4:
                    self._labels_shape.update(
                        {
                            name: (
                                labels[name].shape[1],
                                labels[name].shape[2],
                                labels[name].shape[3],
                            )
                        }
                    )

        print(
            f"Number of Training Data: {self.num_train}, Number of Validation Data: {self.val_num}"
        )

        return input_data, labels

    def pre_testing_checklist_master(self, input_data):
        if type(input_data) is not dict:
            input_data = {self.input_names[0]: np.atleast_2d(input_data)}
        else:
            for name in input_data.keys():
                input_data.update({name: np.atleast_2d(input_data[name])})
        return input_data

    def post_training_checklist_master(self):
        pass

    def save(self, name=None, model_plot=False):
        """
        Save the model to disk

        :param name: Folder name/path to be saved
        :type name: string or path
        :param model_plot: True to plot model too
        :type model_plot: boolean
        :return: A saved folder on disk
        """
        self.has_model_check()
        # Only generate a folder automatically if no name provided
        if self.folder_name is None and name is None:
            self.folder_name = folder_runnum()
        elif name is not None:
            self.folder_name = name
        self.folder_name = pathlib.Path(self.folder_name).absolute()
        # if foldername provided, then create a directory, if exist append something to avoid overwrite
        if not self.folder_name.exists():
            os.makedirs(self.folder_name)
        else:
            i_back = 2
            while True:
                if not self.folder_name.with_name(
                    self.folder_name.stem + f"_{i_back}"
                ).exists():
                    break
                i_back += 1
            new_folder_name_temp = self.folder_name.with_name(
                self.folder_name.stem + f"_{i_back}"
            )
            warnings.warn(
                f"To prevent your model being overwritten, your folder name changed from {self.folder_name} "
                f"to {new_folder_name_temp}",
                UserWarning,
            )
            self.folder_name = new_folder_name_temp
            os.makedirs(self.folder_name)

        self.fullfilepath = str(self.folder_name) + pathlib.os.sep
        txt_file_path = pathlib.Path.joinpath(self.folder_name, "hyperparameter.txt")
        if os.path.isfile(txt_file_path):
            self.hyper_txt = open(txt_file_path, "a")
            self.hyper_txt.write("\n")
            self.hyper_txt.write("======Another Run======")
        else:
            self.hyper_txt = open(txt_file_path, "w")
        self.hyper_txt.write(f"Model: {self.name} \n")
        self.hyper_txt.write(f"Model Type: {self._model_type} \n")
        self.hyper_txt.write(f"astroNN identifier: {self._model_identifier} \n")
        self.hyper_txt.write(f"Python Version: {self._python_info} \n")
        self.hyper_txt.write(f"astroNN Version: {self._astronn_ver} \n")
        self.hyper_txt.write(f"Keras Version: {self._keras_ver} \n")
        self.hyper_txt.write(f"Tensorflow Version: {self._tf_ver} \n")
        self.hyper_txt.write(f"Folder Name: {self.folder_name.name} \n")
        self.hyper_txt.write(f"Batch size: {self.batch_size} \n")
        self.hyper_txt.write(f"Optimizer: {self.optimizer.__class__.__name__} \n")
        self.hyper_txt.write(f"Maximum Epochs: {self.max_epochs} \n")
        self.hyper_txt.write(f"Learning Rate: {self.lr} \n")
        self.hyper_txt.write(f"Validation Size: {self.val_size} \n")
        self.hyper_txt.write(f"Input Shape: {self._input_shape} \n")
        self.hyper_txt.write(f"Label Shape: {self._labels_shape} \n")
        self.hyper_txt.write(f"Number of Training Data: {self.num_train} \n")
        self.hyper_txt.write(f"Number of Validation Data: {self.val_num} \n")

        if model_plot is True:
            self.plot_model()

        self.post_training_checklist_child()

        if (
            self.virtual_cvslogger is not None
        ):  # in case you save without training, so cvslogger is None
            self.virtual_cvslogger.savefile(folder_name=self.folder_name)

    def plot_model(
        self, name="model.png", show_shapes=True, show_layer_names=True, rankdir="TB"
    ):
        """
        Plot model architecture with pydot and graphviz

        :param name: file name to be saved with extension, .png is recommended
        :type name: str
        :param show_shapes: whether show shape in model plot
        :type show_shapes: bool
        :param show_layer_names: whether to display layer names
        :type show_layer_names: bool
        :param rankdir: a string specifying the format of the plot, 'TB' for vertical or 'LR' for horizontal plot
        :type rankdir: bool
        :return: No return but will save the model architecture as png to disk
        """
        self.has_model_check()

        try:
            if self.fullfilepath is not None:
                plot_model(
                    self.keras_model,
                    show_shapes=show_shapes,
                    to_file=os.path.join(self.fullfilepath, name),
                    show_layer_names=show_layer_names,
                    rankdir=rankdir,
                )
            else:
                plot_model(
                    self.keras_model,
                    show_shapes=show_shapes,
                    to_file=name,
                    show_layer_names=show_layer_names,
                    rankdir=rankdir,
                )
        except ImportError or ModuleNotFoundError:
            warnings.warn(
                "Skipped plot_model! graphviz and pydot_ng are required to plot the model architecture",
                UserWarning,
            )
            pass

    def hessian(self, x=None, mean_output=False, mc_num=1, denormalize=False):
        """
        | Calculate the hessian of output to input
        |
        | Please notice that the de-normalize (if True) assumes the output depends on the input data first orderly
        | in which the hessians does not depends on input scaling and only depends on output scaling
        |
        | The hessians can be all zeros and the common cause is you did not use any activation or
        | activation that is still too linear in some sense like ReLU.

        :param x: Input Data
        :type x: ndarray
        :param mean_output: False to get all hessian, True to get the mean
        :type mean_output: boolean
        :param mc_num: Number of monte carlo integration
        :type mc_num: int
        :param denormalize: De-normalize diagonal part of Hessian
        :type denormalize: bool
        :return: An array of Hessian
        :rtype: ndarray
        :History: 2018-Jun-14 - Written - Henry Leung (University of Toronto)
        """
        self.has_model_check()

        if x is None:
            raise ValueError("Please provide data to calculate the jacobian")

        if mc_num < 1 or isinstance(mc_num, float):
            raise ValueError("mc_num must be a positive integer")

        if self.input_normalizer is not None:
            x_data = self.input_normalizer.normalize({"input": x}, calc=False)
            x_data = x_data["input"]
        else:
            # Prevent shallow copy issue
            x_data = np.array(x)
            x_data -= self.input_mean
            x_data /= self.input_std

        _model = None
        try:
            input_tens = self.keras_model_predict.get_layer("input").input
            output_tens = self.keras_model_predict.get_layer("output").output
            input_shape_expectation = self.keras_model_predict.get_layer(
                "input"
            ).input_shape
            output_shape_expectation = self.keras_model_predict.get_layer(
                "output"
            ).output_shape
            _model = self.keras_model_predict
        except AttributeError:
            input_tens = self.keras_model.get_layer("input").input
            output_tens = self.keras_model.get_layer("output").output
            input_shape_expectation = self.keras_model.get_layer("input").input_shape
            output_shape_expectation = self.keras_model.get_layer("output").output_shape
            _model = self.keras_model
        except ValueError:
            raise ValueError(
                "astroNN expects input layer is named as 'input' and output layer is named as 'output', "
                "but None is found."
            )

        if len(input_shape_expectation) == 1:
            input_shape_expectation = input_shape_expectation[0]

        # just in case only 1 data point is provided and mess up the shape issue
        if len(input_shape_expectation) == 3:
            x_data = np.atleast_3d(x_data)
        elif len(input_shape_expectation) == 4:
            if len(x_data.shape) < 4:
                x_data = x_data[:, :, :, np.newaxis]
        else:
            raise ValueError("Input data shape do not match neural network expectation")

        total_num = x_data.shape[0]

        input_dim = len(np.squeeze(np.ones(input_shape_expectation[1:])).shape)
        output_dim = len(np.squeeze(np.ones(output_shape_expectation[1:])).shape)
        if input_dim > 3 or output_dim > 3:
            raise ValueError("Unsupported data dimension")

        xtensor = tf.Variable(x_data)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(xtensor)
            with tf.GradientTape() as dtape:
                dtape.watch(xtensor)
                temp = _model(xtensor)
            jacobian = tf.squeeze(dtape.batch_jacobian(temp, xtensor))

        start_time = time.time()

        hessian = tf.squeeze(tape.batch_jacobian(jacobian, xtensor))

        if np.all(
            hessian == 0.0
        ):  # warn user about not so linear activation like ReLU will get all zeros
            warnings.warn(
                "The hessians is detected to be all zeros. The common cause is you did not use any activation or "
                "activation that is still too linear in some sense like ReLU.",
                UserWarning,
            )

        if mean_output is True:
            hessians_master = tf.reduce_mean(hessian, axis=0).numpy()
        else:
            hessians_master = hessian.numpy()

        if (
            denormalize
        ):  # no need to denorm input scaling because of we assume first order dependence
            if self.labels_std is not None:
                try:
                    hessians_master = hessians_master * self.labels_std
                except ValueError:
                    hessians_master = hessians_master * self.labels_std.reshape(-1, 1)

        print(
            f"Finished hessian calculation, {(time.time() - start_time):.{2}f} seconds elapsed"
        )
        return hessians_master

    def jacobian(self, x=None, mean_output=False, mc_num=1, denormalize=False):
        """
        | Calculate jacobian of gradient of output to input high performance calculation update on 15 April 2018
        |
        | Please notice that the de-normalize (if True) assumes the output depends on the input data first orderly
        | in which the equation is simply jacobian divided the input scaling, usually a good approx. if you use ReLU all the way

        :param x: Input Data
        :type x: ndarray
        :param mean_output: False to get all jacobian, True to get the mean
        :type mean_output: boolean
        :param mc_num: Number of monte carlo integration
        :type mc_num: int
        :param denormalize: De-normalize Jacobian
        :type denormalize: bool
        :return: An array of Jacobian
        :rtype: ndarray
        :History:
            | 2017-Nov-20 - Written - Henry Leung (University of Toronto)
            | 2018-Apr-15 - Updated - Henry Leung (University of Toronto)
        """
        self.has_model_check()
        if x is None:
            raise ValueError("Please provide data to calculate the jacobian")

        if mc_num < 1 or isinstance(mc_num, float):
            raise ValueError("mc_num must be a positive integer")

        if self.input_normalizer is not None:
            x_data = self.input_normalizer.normalize({"input": x}, calc=False)
            x_data = x_data["input"]
        else:
            # Prevent shallow copy issue
            x_data = np.array(x)
            x_data -= self.input_mean
            x_data /= self.input_std

        _model = None
        try:
            input_tens = self.keras_model_predict.get_layer("input").input
            output_tens = self.keras_model_predict.get_layer("output").output
            input_shape_expectation = self.keras_model_predict.get_layer(
                "input"
            ).input_shape
            output_shape_expectation = self.keras_model_predict.get_layer(
                "output"
            ).output_shape
            _model = self.keras_model_predict
        except AttributeError:
            input_tens = self.keras_model.get_layer("input").input
            output_tens = self.keras_model.get_layer("output").output
            input_shape_expectation = self.keras_model.get_layer("input").input_shape
            output_shape_expectation = self.keras_model.get_layer("output").output_shape
            _model = self.keras_model
        except ValueError:
            raise ValueError(
                "astroNN expects input layer is named as 'input' and output layer is named as 'output', "
                "but None is found."
            )

        if len(input_shape_expectation) == 1:
            input_shape_expectation = input_shape_expectation[0]

        # just in case only 1 data point is provided and mess up the shape issue
        if len(input_shape_expectation) == 3:
            x_data = np.atleast_3d(x_data)
        elif len(input_shape_expectation) == 4:
            if len(x_data.shape) < 4:
                x_data = x_data[:, :, :, np.newaxis]
        else:
            raise ValueError("Input data shape do not match neural network expectation")

        total_num = x_data.shape[0]

        # TODO: move this to master??
        input_dim = len(np.squeeze(np.ones(input_shape_expectation[1:])).shape)
        output_dim = len(np.squeeze(np.ones(output_shape_expectation[1:])).shape)
        if input_dim > 3 or output_dim > 3:
            raise ValueError("Unsupported data dimension")

        xtensor = tf.Variable(x_data)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(xtensor)
            temp = _model(xtensor)

        start_time = time.time()

        jacobian = tf.squeeze(tape.batch_jacobian(temp, xtensor))

        if mean_output is True:
            jacobian_master = tf.reduce_mean(jacobian, axis=0).numpy()
        else:
            jacobian_master = jacobian.numpy()

        if denormalize:
            if self.input_std is not None:
                jacobian_master = jacobian_master / np.squeeze(self.input_std)

            if self.labels_std is not None:
                try:
                    jacobian_master = jacobian_master * self.labels_std
                except ValueError:
                    jacobian_master = jacobian_master * self.labels_std.reshape(-1, 1)

        print(
            f"Finished all gradient calculation, {(time.time() - start_time):.{2}f} seconds elapsed"
        )

        return jacobian_master

    def plot_dense_stats(self):
        """
        Plot dense layers weight statistics

        :return: A plot
        :History: 2018-May-12 - Written - Henry Leung (University of Toronto)
        """
        self.has_model_check()
        dense_list = []
        for counter, layer in enumerate(self.keras_model.layers):
            if isinstance(layer, tfk.layers.Dense):
                dense_list.append(counter)

        denses = np.array(self.keras_model.layers)[dense_list]
        fig, ax = plt.subplots(1, figsize=(15, 10), dpi=100)
        for counter, dense in enumerate(denses):
            weight_temp = np.array(dense.get_weights()[0].flatten())
            ax.hist(
                weight_temp,
                200,
                density=True,
                range=(-2.0, 2.0),
                alpha=0.7,
                label=f"Dense Layer {counter}, max: {weight_temp.max():.{2}f}, min: {weight_temp.min():.{2}f}, "
                f"mean: {weight_temp.mean():.{2}f}, std: {weight_temp.std():.{2}f}",
            )
        fig.suptitle(
            f"Dense Layers Weight Statistics of {self.folder_name}", fontsize=17
        )
        ax.set_xlabel("Weights", fontsize=17)
        ax.set_ylabel("Normalized Distribution", fontsize=17)
        ax.minorticks_on()
        ax.tick_params(labelsize=15, width=3, length=10, which="major")
        ax.tick_params(width=1.5, length=5, which="minor")
        ax.legend(loc="best", fontsize=15)
        fig.tight_layout(rect=[0, 0.00, 1, 0.96])
        fig.show()

        return fig

    @property
    def output_shape(self):
        """
        Get output shape of the prediction model

        :return: output shape expectation
        :rtype: tuple
        :History: 2018-May-19 - Written - Henry Leung (University of Toronto)
        """
        self.has_model_check()
        try:
            return self.keras_model_predict.output_shape
        except AttributeError:
            return self.keras_model.output_shape

    @property
    def input_shape(self):
        """
        Get input shape of the prediction model

        :return: input shape expectation
        :rtype: tuple
        :History: 2018-May-21 - Written - Henry Leung (University of Toronto)
        """
        self.has_model_check()
        try:
            return self.keras_model_predict.input_shape
        except AttributeError:
            return self.keras_model.input_shape

    def get_weights(self):
        """
        Get all model weights

        :return: weights arrays
        :rtype: ndarray
        :History: 2018-May-23 - Written - Henry Leung (University of Toronto)
        """
        self.has_model_check()
        return self.keras_model.get_weights()

    def summary(self):
        """
        Get model summary

        :return: None, just print
        :History: 2018-May-23 - Written - Henry Leung (University of Toronto)
        """
        self.has_model_check()
        return self.keras_model.summary()

    def get_config(self):
        """
        Get model configuration as a dictionary

        :return: dict
        :History: 2018-May-23 - Written - Henry Leung (University of Toronto)
        """
        self.has_model_check()
        return self.keras_model.get_config()

    def save_weights(self, filename=_astroNN_MODEL_NAME, overwrite=True):
        """
        Save model weights as .h5

        :param filename: Filename of .h5 to be saved
        :type filename: str
        :param overwrite: whether to overwrite
        :type overwrite: bool
        :return: None, a .h5 file will be saved
        :History: 2018-May-23 - Written - Henry Leung (University of Toronto)
        """
        self.has_model_check()
        print("==========================")
        print(
            "This is a remainder that saving weights to h5, you might have difficult to "
            "load it back and cannot be used with astroNN probably"
        )
        print("==========================")
        if self.fullfilepath is not None:
            return self.keras_model.save_weights(
                str(os.path.join(self.fullfilepath, filename)), overwrite=overwrite
            )
        else:
            return self.keras_model.save_weights(filename, overwrite=overwrite)

    @property
    def uses_learning_phase(self):
        """
        To determine whether the model depends on keras learning flag. If False, then setting learning phase will not
        affect the model

        :return: the boolean to indicate keras learning flag dependence of the model
        :rtype: bool
        :History: 2018-Jun-03 - Written - Henry Leung (University of Toronto)
        """
        self.has_model_check()
        return any(
            [
                getattr(x, "_uses_learning_phase", False)
                for x in self.keras_model.outputs
            ]
        )

    def get_layer(self, *args, **kwargs):
        """
        get_layer() method of tensorflow
        """
        return self.keras_model.get_layer(*args, **kwargs)

    def flush(self):
        """
        | Experimental, I don't think it works
        | Flush GPU memory from tensorflow

        :History: 2018-Jun-19 - Written - Henry Leung (University of Toronto)
        """
        tfk.backend.clear_session()

    def transfer_weights(self, model, exclusion_output=False):
        """
        Transfer weight of a model to current model if possible
        # TODO: remove layers after successful transfer so wont mix up?

        :param model: astroNN model
        :type model: astroNN.model.NeuralNetMaster or keras.models.Model
        :param exclusion_output: whether to exclude output in the transfer or not
        :type exclusion_output: bool
        :return: bool
        :History: 2022-Mar-06 - Written - Henry Leung (University of Toronto)
        """

        if hasattr(
            model, "keras_model"
        ):  # check if its an astroNN model or keras model
            model = model.keras_model

        counter = 0  # count number of weights transferred
        transferred = []  # keep track of transferred layer names
        total_parameters_A = count_params(self.keras_model.weights)
        total_parameters_B = count_params(model.weights)
        current_bottom_idx = 0  # current bottom layer we are checking to prevent incorrect transfer of convolution layer weights

        for new_l in self.keras_model.layers:
            for idx, l in enumerate(model.layers[current_bottom_idx:]):
                if not "input" in l.name and not "input" in new_l.name:  # no need to do
                    try:
                        if (not "output" in l.name or not exclusion_output) and len(
                            new_l.get_weights()
                        ) != 0:
                            new_l.set_weights(l.get_weights())
                            new_l.trainable = False
                            for i in l.get_weights():
                                counter += len(tf.reshape(i, [-1]))
                            transferred.append(l.name)
                            current_bottom_idx += idx
                        break
                    except ValueError:
                        pass

        if counter == 0:
            warnings.warn(
                "None of the layers' weights are successfully transfered due to shape incompatibility in all layers."
            )
        else:
            self.recompile()
            print(f"Successfully transferred: {transferred}")
            print(
                f"Transferred {counter} of {total_parameters_B} weights ({100*counter/total_parameters_B:.2f}%) to a new model with {total_parameters_A} weights."
            )
