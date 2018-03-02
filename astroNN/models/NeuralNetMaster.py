###############################################################################
#   NeuralNetMaster.py: top-level class for a neural network
###############################################################################
import os
import sys
import time
from abc import ABC, abstractmethod

import keras
import tensorflow as tf
from keras.backend import get_session, epsilon
from keras.utils import plot_model

import astroNN
from astroNN.shared.nn_tools import folder_runnum, cpu_fallback, gpu_memory_manage


class NeuralNetMaster(ABC):
    """Top-level class for a neural network"""

    def __init__(self):
        """
        NAME:
            __init__
        PURPOSE:
            To define astroNN neural network
        HISTORY:
            2017-Dec-23 - Written - Henry Leung (University of Toronto)
            2018-Jan-05 - Update - Henry Leung (University of Toronto)
        """
        self.name = None
        self._model_type = None
        self._model_identifier = None
        self._implementation_version = None
        self.__python_info = sys.version
        self.__astronn_ver = astroNN.__version__
        self.__keras_ver = keras.__version__
        self.__tf_ver = tf.VERSION
        self.fallback_cpu = False
        self.limit_gpu_mem = True
        self.log_device_placement = False
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
        self.val_num = None

        # optimizer parameter
        self.beta_1 = 0.9  # exponential decay rate for the 1st moment estimates for optimization algorithm
        self.beta_2 = 0.999  # exponential decay rate for the 2nd moment estimates for optimization algorithm
        self.optimizer_epsilon = epsilon()  # a small constant for numerical stability for optimization algorithm
        self.optimizer = None

        # Keras API
        self.verbose = 2
        self.keras_model = None
        self.keras_model_predict = None
        self.history = None
        self.metrics = None

        self.input_normalizer = None
        self.labels_normalizer = None
        self.training_generator = None

        self.input_norm_mode = None
        self.labels_norm_mode = None
        self.input_mean_norm = None
        self.input_std_norm = None
        self.labels_mean_norm = None
        self.labels_std_norm = None

        self.input_shape = None
        self.labels_shape = None

        self.num_train = None
        self.targetname = None
        self.history = None
        self.virtual_cvslogger = None

    @abstractmethod
    def train(self, *args):
        raise NotImplementedError

    @abstractmethod
    def test(self, *args):
        raise NotImplementedError

    @abstractmethod
    def model(self):
        raise NotImplementedError

    @abstractmethod
    def post_training_checklist_child(self):
        raise NotImplementedError

    def cpu_gpu_check(self):
        if self.fallback_cpu is True:
            cpu_fallback()

        if self.limit_gpu_mem is True:
            gpu_memory_manage(log_device_placement=self.log_device_placement)
        elif isinstance(self.limit_gpu_mem, float) is True:
            gpu_memory_manage(ratio=self.limit_gpu_mem, log_device_placement=self.log_device_placement)

    def pre_training_checklist_master(self, input_data, labels):
        if self.val_size is None:
            self.val_size = 0
        self.val_num = int(input_data.shape[0] * self.val_size)
        self.num_train = input_data.shape[0] - self.val_num

        if input_data.ndim == 2:
            self.input_shape = (input_data.shape[1], 1,)
        elif input_data.ndim == 3:
            self.input_shape = (input_data.shape[1], input_data.shape[2], 1,)
        elif input_data.ndim == 4:
            self.input_shape = (input_data.shape[1], input_data.shape[2], input_data.shape[3],)

        if labels.ndim == 1:
            self.labels_shape = 1
        elif labels.ndim == 2:
            self.labels_shape = labels.shape[1]
        elif labels.ndim == 3:
            self.labels_shape = (labels.shape[1], labels.shape[2])
        elif labels.ndim == 4:
            self.labels_shape = (labels.shape[1], labels.shape[2], labels.shape[3])

        self.cpu_gpu_check()

        print('Number of Training Data: {}, Number of Validation Data: {}'.format(self.num_train, self.val_num))

    def pre_testing_checklist_master(self):
        pass

    def post_training_checklist_master(self):
        pass

    def save(self, name=None, model_plot=False):
        # Only generate a folder automatically if no name provided
        if self.folder_name is None and name is None:
            self.folder_name = folder_runnum()
        else:
            if name is not None:
                self.folder_name = name
        # if foldername provided, then create a directory
        if not os.path.exists(os.path.join(self.currentdir, self.folder_name)):
            os.makedirs(os.path.join(self.currentdir, self.folder_name))

        self.fullfilepath = os.path.join(self.currentdir, self.folder_name + '/')

        self.hyper_txt = open(self.fullfilepath + 'hyperparameter.txt', 'w')
        self.hyper_txt.write("Model: {} \n".format(self.name))
        self.hyper_txt.write("Model Type: {} \n".format(self._model_type))
        self.hyper_txt.write("astroNN identifier: {} \n".format(self._model_identifier))
        self.hyper_txt.write("Python Version: {} \n".format(self.__python_info))
        self.hyper_txt.write("astroNN Version: {} \n".format(self.__astronn_ver))
        self.hyper_txt.write("Keras Version: {} \n".format(self.__keras_ver))
        self.hyper_txt.write("Tensorflow Version: {} \n".format(self.__tf_ver))
        self.hyper_txt.write("Folder Name: {} \n".format(self.folder_name))
        self.hyper_txt.write("Fallback CPU? : {} \n".format(self.fallback_cpu))
        self.hyper_txt.write("astroNN GPU Management: {} \n".format(self.limit_gpu_mem))
        self.hyper_txt.write("Batch size: {} \n".format(self.batch_size))
        self.hyper_txt.write("Optimizer: {} \n".format(self.optimizer.__class__.__name__))
        self.hyper_txt.write("Maximum Epochs: {} \n".format(self.max_epochs))
        self.hyper_txt.write("Learning Rate: {} \n".format(self.lr))
        self.hyper_txt.write("Validation Size: {} \n".format(self.val_size))
        self.hyper_txt.write("Input Shape: {} \n".format(self.input_shape))
        self.hyper_txt.write("Label Shape: {} \n".format(self.labels_shape))
        self.hyper_txt.write("Number of Training Data: {} \n".format(self.num_train))
        self.hyper_txt.write("Number of Validation Data: {} \n".format(self.val_num))

        if model_plot is True:
            self.plot_model()

        self.post_training_checklist_child()

        self.virtual_cvslogger.savefile(folder_name=self.folder_name)

    def plot_model(self):
        try:
            if self.fullfilepath is not None:
                plot_model(self.keras_model, show_shapes=True, to_file=self.fullfilepath + 'model.png')
            else:
                plot_model(self.keras_model, show_shapes=True, to_file='model.png')
        except Exception:
            print('Skipped plot_model! graphviz and pydot_ng are required to plot the model architecture')
            pass

    def jacobian(self, x=None, mean_output=False):
        """
        NAME: jacobian
        PURPOSE: calculate jacobian of gradietn of output to input
        INPUT:
            x (ndarray): Input Data
            mean_output (boolean): False to get all jacobian, True to get the mean
        OUTPUT:
            (ndarray): Jacobian
        HISTORY:
            2017-Nov-20 Henry Leung
        """
        import numpy as np

        if x is None:
            raise ValueError('Please provide data to calculate the jacobian')

        x_data = np.array(x)
        x_data -= self.input_mean_norm
        x_data /= self.input_std_norm

        get_session().run(tf.global_variables_initializer())

        try:
            input_tens = self.keras_model_predict.get_layer("input").input
            input_shape_expectation = self.keras_model_predict.get_layer("input").input_shape
        except AttributeError:
            input_tens = self.keras_model.get_layer("input").input
            input_shape_expectation = self.keras_model.get_layer("input").input_shape

        start_time = time.time()

        if len(input_shape_expectation) == 3:
            x_data = np.atleast_3d(x_data)

            grad_list = []
            for j in range(self.labels_shape):
                grad_list.append(tf.gradients(self.keras_model.get_layer("output").output[0, j], input_tens))

            final_stack = tf.stack(tf.squeeze(grad_list))
            jacobian = np.ones((self.labels_shape, x_data.shape[1], x_data.shape[0]), dtype=np.float32)

            for i in range(x_data.shape[0]):
                x_in = x_data[i:i + 1]
                jacobian[:, :, i] = get_session().run(final_stack, feed_dict={input_tens: x_in})

        elif len(input_shape_expectation) == 4:
            monoflag = False
            if len(x_data.shape) < 4:
                monoflag = True
                x_data = x_data[:, :, :, np.newaxis]

            jacobian = np.ones((self.labels_shape, x_data.shape[2], x_data.shape[1], x_data.shape[3], x_data.shape[0]),
                               dtype=np.float32)

            grad_list = []
            for j in range(self.labels_shape):
                grad_list.append(tf.gradients(self.keras_model.get_layer("output").output[0, j], input_tens))

            final_stack = tf.stack(tf.squeeze(grad_list))

            for i in range(x_data.shape[0]):
                x_in = x_data[i:i + 1]
                if monoflag is False:
                    jacobian[:, :, :, :, i] = get_session().run(final_stack, feed_dict={input_tens: x_in})
                else:
                    jacobian[:, :, :, 0, i] = get_session().run(final_stack, feed_dict={input_tens: x_in})

        else:
            raise ValueError('Input Data shape do not match neural network expectation')

        if mean_output is True:
            jacobian = np.mean(jacobian, axis=-1)

        print('Finished gradient calculation, {:.03f} seconds elapsed'.format(time.time() - start_time))

        return jacobian
