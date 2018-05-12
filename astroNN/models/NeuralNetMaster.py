###############################################################################
#   NeuralNetMaster.py: top-level class for a neural network
###############################################################################
import os
import sys
import time
from abc import ABC, abstractmethod

import numpy as np
import pylab as plt
import tensorflow as tf

import astroNN
from astroNN.config import keras_import_manager, cpu_gpu_check
from astroNN.shared.nn_tools import folder_runnum

keras = keras_import_manager()
get_session, epsilon, plot_model = keras.backend.get_session, keras.backend.epsilon, keras.utils.plot_model


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
    :ivar beta_2: Eexponential decay rate for the 2nd moment estimates for optimization algorithm
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
        self._keras_ver = keras.__version__  # Even using tensorflow.keras, this line will still be fine
        self._tf_ver = tf.VERSION
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
        self.callbacks = None
        self.__callbacks = None  # for internal default callbacks usage only

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

        self.input_shape = None
        self.labels_shape = None

        self.num_train = None
        self.train_idx = None
        self.val_idx = None
        self.targetname = None
        self.history = None
        self.virtual_cvslogger = None
        self.hyper_txt = None

        cpu_gpu_check()

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

    def pre_training_checklist_master(self, input_data, labels):
        if self.val_size is None:
            self.val_size = 0
        self.val_num = int(input_data.shape[0] * self.val_size)
        self.num_train = input_data.shape[0] - self.val_num

        # Assuming the convolutional layer immediately after input layer
        # only require if it is new, no need for fine-tuning
        if self.input_shape is None:
            if input_data.ndim == 1:
                self.input_shape = (1, 1,)
            elif input_data.ndim == 2:
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

        print(f'Number of Training Data: {self.num_train}, Number of Validation Data: {self.val_num}')

    def pre_testing_checklist_master(self):
        pass

    def post_training_checklist_master(self):
        pass

    def save(self, name=None, model_plot=False):
        """
        Save the model to disk

        :param name: Folder name to be saved
        :type name: string
        :param model_plot: True to plot model too
        :type model_plot: boolean
        :return: A saved folder on disk
        """
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

        txt_file_path = self.fullfilepath + 'hyperparameter.txt'
        if os.path.isfile(txt_file_path):
            self.hyper_txt = open(txt_file_path, 'a')
            self.hyper_txt.write("\n")
            self.hyper_txt.write("======Another Run======")
        else:
            self.hyper_txt = open(txt_file_path, 'w')
        self.hyper_txt.write(f"Model: {self.name} \n")
        self.hyper_txt.write(f"Model Type: {self._model_type} \n")
        self.hyper_txt.write(f"astroNN identifier: {self._model_identifier} \n")
        self.hyper_txt.write(f"Python Version: {self._python_info} \n")
        self.hyper_txt.write(f"astroNN Version: {self._astronn_ver} \n")
        self.hyper_txt.write(f"Keras Version: {self._keras_ver} \n")
        self.hyper_txt.write(f"Tensorflow Version: {self._tf_ver} \n")
        self.hyper_txt.write(f"Folder Name: {self.folder_name} \n")
        self.hyper_txt.write(f"Batch size: {self.batch_size} \n")
        self.hyper_txt.write(f"Optimizer: {self.optimizer.__class__.__name__} \n")
        self.hyper_txt.write(f"Maximum Epochs: {self.max_epochs} \n")
        self.hyper_txt.write(f"Learning Rate: {self.lr} \n")
        self.hyper_txt.write(f"Validation Size: {self.val_size} \n")
        self.hyper_txt.write(f"Input Shape: {self.input_shape} \n")
        self.hyper_txt.write(f"Label Shape: {self.labels_shape} \n")
        self.hyper_txt.write(f"Number of Training Data: {self.num_train} \n")
        self.hyper_txt.write(f"Number of Validation Data: {self.val_num} \n")

        if model_plot is True:
            self.plot_model()

        self.post_training_checklist_child()

        self.virtual_cvslogger.savefile(folder_name=self.folder_name)

    def plot_model(self):
        """
        Plot model architecture

        :return: No return but will save the model architecture as png to disk
        """
        try:
            if self.fullfilepath is not None:
                plot_model(self.keras_model, show_shapes=True, to_file=self.fullfilepath + 'model.png')
            else:
                plot_model(self.keras_model, show_shapes=True, to_file='model.png')
        except ImportError or ModuleNotFoundError:
            print('Skipped plot_model! graphviz and pydot_ng are required to plot the model architecture')
            pass

    def jacobian(self, x=None, mean_output=False, batch_size=64, mc_num=1):
        """
        Calculate jacobian of gradietn of output to input high performance calculation update on 15 April 2018

        :param x: Input Data
        :type x: ndarray
        :param mean_output: False to get all jacobian, True to get the mean
        :type mean_output: boolean
        :param batch_size: Batch size used to calculate jacobian
        :type batch_size: int
        :param mc_num: Number of monte carlo integration
        :type mc_num: int
        :return: An array of Jacobian
        :rtype: ndarray
        :History:
            | 2017-Nov-20 - Written - Henry Leung (University of Toronto)
            | 2018-Apr-15 - Updated - Henry Leung (University of Toronto)
        """
        if x is None:
            raise ValueError('Please provide data to calculate the jacobian')

        if mc_num < 1 or isinstance(mc_num, float):
            raise ValueError('mc_num must be a positive integer')

        if batch_size < 1 or isinstance(batch_size, float):
            raise ValueError('batch_size must be a positive integer')

        if self.input_normalizer is not None:
            x_data = self.input_normalizer.normalize(x, calc=False)
        else:
            # Prevent shallow copy issue
            x_data = np.array(x)
            x_data -= self.input_mean
            x_data /= self.input_std

        try:
            input_tens = self.keras_model_predict.get_layer("input").input
            output_tens = self.keras_model_predict.get_layer("output").output
            input_shape_expectation = self.keras_model_predict.get_layer("input").input_shape
            output_shape_expectation = self.keras_model_predict.get_layer("output").output_shape
        except AttributeError:
            input_tens = self.keras_model.get_layer("input").input
            output_tens = self.keras_model.get_layer("output").output
            input_shape_expectation = self.keras_model.get_layer("input").input_shape
            output_shape_expectation = self.keras_model.get_layer("output").output_shape

        # just in case only 1 data point is provided and mess up the shape issue
        if len(input_shape_expectation) == 3:
            x_data = np.atleast_3d(x_data)
        elif len(input_shape_expectation) == 4:
            if len(x_data.shape) < 4:
                x_data = x_data[:, :, :, np.newaxis]
        else:
            raise ValueError('Input data shape do not match neural network expectation')

        total_num = x_data.shape[0]
        # if batch_size > total_num, then we do all inputs at once
        if total_num < batch_size:
            batch_size = total_num

        grad_list = []
        for j in range(self.labels_shape):
            grad_list.append(tf.gradients(output_tens[:, j], input_tens))

        final_stack = tf.stack(tf.squeeze(grad_list))

        # Looping variables for tensorflow setup
        i = tf.constant(0)
        mc_num_tf = tf.constant(mc_num)
        #  To store final result
        l = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=1, dynamic_size=True)

        def body(i, l):
            l = l.write(i, final_stack)
            return i + 1, l

        tf_index, loop = tf.while_loop(lambda i, *_: tf.less(i, mc_num_tf), body, [i, l])

        loops = tf.cond(tf.greater(mc_num_tf, 1), lambda: tf.reduce_mean(loop.stack(), axis=0), lambda: loop.stack())
        loops = tf.reshape(loops, shape=[tf.shape(input_tens)[0], *output_shape_expectation[1:], *input_shape_expectation[1:]])
        start_time = time.time()

        jacobian = np.concatenate([get_session().run(loops, feed_dict={input_tens: x_data[i:i+batch_size]}) for i in
                                   range(0, total_num, batch_size)], axis=0)

        if mean_output is True:
            jacobian_master = np.mean(jacobian, axis=0)
        else:
            jacobian_master = np.array(jacobian)

        print(f'Finished gradient calculation, {(time.time() - start_time):.{2}f} seconds elapsed')

        return np.squeeze(jacobian_master)

    def jacobian_old(self, x=None, mean_output=False):
        """
        Calculate jacobian of gradietn of output to input

        :param x: Input Data
        :type x: ndarray
        :param mean_output: False to get all jacobian, True to get the mean
        :type mean_output: boolean
        :History: 2017-Nov-20 - Written - Henry Leung (University of Toronto)
        """
        if x is None:
            raise ValueError('Please provide data to calculate the jacobian')

        if self.input_normalizer is not None:
            x_data = self.input_normalizer.normalize(x, calc=False)
        else:
            # Prevent shallow copy issue
            x_data = np.array(x)
            x_data -= self.input_mean
            x_data /= self.input_std

        try:
            input_tens = self.keras_model_predict.get_layer("input").input
            output_tens = self.keras_model_predict.get_layer("output").output
            input_shape_expectation = self.keras_model_predict.get_layer("input").input_shape
        except AttributeError:
            input_tens = self.keras_model.get_layer("input").input
            output_tens = self.keras_model.get_layer("output").output
            input_shape_expectation = self.keras_model.get_layer("input").input_shape

        start_time = time.time()

        if len(input_shape_expectation) == 3:
            x_data = np.atleast_3d(x_data)

            grad_list = []
            for j in range(self.labels_shape):
                grad_list.append(tf.gradients(output_tens[0, j], input_tens))

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

            jacobian = np.ones((self.labels_shape, x_data.shape[1], x_data.shape[2], x_data.shape[3], x_data.shape[0]),
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
            raise ValueError('Input data shape do not match neural network expectation')

        if mean_output is True:
            jacobian = np.mean(jacobian, axis=-1)

        print(f'Finished gradient calculation, {(time.time() - start_time):.{2}f} seconds elapsed')

        return jacobian

    def plot_dense_stats(self):
        """
        Plot Dense Layers Weight Statistics

        :return: A plot
        :History: 2018-May-12 - Written - Henry Leung (University of Toronto)
        """
        dense_list = []
        for counter, layer in enumerate(self.keras_model.layers):
            if isinstance(layer, keras.layers.Dense):
                dense_list.append(counter)

        denses = np.array(self.keras_model.layers)[dense_list]
        fig, ax = plt.subplots(1, figsize=(15, 10), dpi=100)
        for counter, dense in enumerate(denses):
            weight_temp = np.array(dense.get_weights())[0].flatten()
            ax.hist(weight_temp, 200, normed=True, range=(-2., 2.), label=f'Dense Layer {counter}, '
                                                                          f'max: {weight_temp.max():.{2}f}, '
                                                                          f'min: {weight_temp.min():.{2}f}, '
                                                                          f'mean: {weight_temp.mean():.{2}f}, '
                                                                          f'std: {weight_temp.std():.{2}f}')
        fig.suptitle(f'Dense Layers Weight Statistics of {self.folder_name}', fontsize=20)
        ax.set_xlabel('Weights', fontsize=15)
        ax.set_ylabel('Normalized Distribution', fontsize=15)
        ax.tick_params(labelsize=15, width=2, length=5, which='major')
        ax.legend(loc='best', fontsize=15)
        fig.tight_layout(rect=[0, 0.00, 1, 0.96])
        fig.show()
