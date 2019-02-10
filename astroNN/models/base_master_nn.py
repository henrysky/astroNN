###############################################################################
#   base_master_nn.py: top-level class for a neural network
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
from astroNN.shared.custom_warnings import deprecated
from astroNN.shared.nn_tools import folder_runnum
from astroNN.config import _astroNN_MODEL_NAME

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

        self._input_shape = None
        self._labels_shape = None

        self.num_train = None
        self.train_idx = None
        self.val_idx = None
        self.targetname = None
        self.history = None
        self.virtual_cvslogger = None
        self.hyper_txt = None

        self.session = None
        self.graph = None

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
            raise AttributeError("No model found in this instance, the common problem is you did not train a model")

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

    def pre_training_checklist_master(self, input_data, labels):
        if self.val_size is None:
            self.val_size = 0
        self.val_num = int(input_data.shape[0] * self.val_size)
        self.num_train = input_data.shape[0] - self.val_num

        # Assuming the convolutional layer immediately after input layer
        # only require if it is new, no need for fine-tuning
        # in case you read this for dense network, use Flattener as first layer in your network to flatten it
        if self._input_shape is None:
            if input_data.ndim == 1:
                self._input_shape = (1, 1,)
            elif input_data.ndim == 2:
                self._input_shape = (input_data.shape[1], 1,)
            elif input_data.ndim == 3:
                self._input_shape = (input_data.shape[1], input_data.shape[2], 1,)
            elif input_data.ndim == 4:
                self._input_shape = (input_data.shape[1], input_data.shape[2], input_data.shape[3],)

            # zeroth dim should always be number of data
            if labels.ndim == 1:
                self._labels_shape = 1
            elif labels.ndim == 2:
                self._labels_shape = (labels.shape[1])
            elif labels.ndim == 3:
                self._labels_shape = (labels.shape[1], labels.shape[2])
            elif labels.ndim == 4:
                self._labels_shape = (labels.shape[1], labels.shape[2], labels.shape[3])

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
        self.has_model_check()
        # Only generate a folder automatically if no name provided
        if self.folder_name is None and name is None:
            self.folder_name = folder_runnum()
        elif name is not None:
            self.folder_name = name

        # if foldername provided, then create a directory, if exist append something to avoid overwrite
        if not os.path.exists(os.path.join(self.currentdir, self.folder_name)):
            os.makedirs(os.path.join(self.currentdir, self.folder_name))
        else:
            i_back = 2
            while True:
                if not os.path.exists(os.path.join(self.currentdir, self.folder_name + f'_{i_back}')):
                    break
                i_back += 1
            new_folder_name_temp = self.folder_name + f'_{i_back}'
            print(f'To prevent your model being overwritten, your folder name changed from {self.folder_name} '
                  f'to {new_folder_name_temp}')
            self.folder_name = new_folder_name_temp
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
        self.hyper_txt.write(f"Input Shape: {self._input_shape} \n")
        self.hyper_txt.write(f"Label Shape: {self._labels_shape} \n")
        self.hyper_txt.write(f"Number of Training Data: {self.num_train} \n")
        self.hyper_txt.write(f"Number of Validation Data: {self.val_num} \n")

        if model_plot is True:
            self.plot_model()

        self.post_training_checklist_child()

        if self.virtual_cvslogger is not None:  # in case you save without training, so cvslogger is None
            self.virtual_cvslogger.savefile(folder_name=self.folder_name)

    def plot_model(self, name='model.png', show_shapes=True, show_layer_names=True, rankdir='TB'):
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
        print()
        try:
            if self.fullfilepath is not None:
                plot_model(self.keras_model, show_shapes=show_shapes, to_file=os.path.join(self.fullfilepath, name),
                           show_layer_names=show_layer_names, rankdir=rankdir)
            else:
                plot_model(self.keras_model, show_shapes=show_shapes, to_file=name, show_layer_names=show_layer_names,
                           rankdir=rankdir)
        except ImportError or ModuleNotFoundError:
            print('Skipped plot_model! graphviz and pydot_ng are required to plot the model architecture')
            pass

    def hessian(self, x=None, mean_output=False, mc_num=1, denormalize=False, method='exact'):
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
        :param method: Either 'exact' to calculate numerical Hessian or 'approx' to approximate Hessian from Jacobian
        :type method: str
        :return: An array of Hessian
        :rtype: ndarray
        :History: 2018-Jun-14 - Written - Henry Leung (University of Toronto)
        """
        if not mean_output:
            print('only mean output is supported at this moment')
            mean_output = True
        if method == 'approx':
            all_args = locals()
            # remove unnecessary argument
            all_args.pop('self')
            all_args.pop('method')
            jacobian = self.jacobian(**all_args)
            hessians_master = np.stack([np.dot(jacobian[x_shape:x_shape + 1].T, jacobian[x_shape:x_shape + 1])
                                        for x_shape in range(jacobian.shape[0])], axis=0)
            return hessians_master

        elif method == 'exact':
            self.has_model_check()
            if x is None:
                raise ValueError('Please provide data to calculate the jacobian')

            if mc_num < 1 or isinstance(mc_num, float):
                raise ValueError('mc_num must be a positive integer')

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
            except ValueError:
                raise ValueError(
                    "astroNN expects input layer is named as 'input' and output layer is named as 'output', "
                    "but None is found.")

            # just in case only 1 data point is provided and mess up the shape issue
            if len(input_shape_expectation) == 3:
                x_data = np.atleast_3d(x_data)
            elif len(input_shape_expectation) == 4:
                if len(x_data.shape) < 4:
                    x_data = x_data[:, :, :, np.newaxis]
            else:
                raise ValueError('Input data shape do not match neural network expectation')

            total_num = x_data.shape[0]

            hessians_list = []
            for j in range(self._labels_shape):
                hessians_list.append(tf.hessians(output_tens[:, j], input_tens))

            final_stack = tf.stack(tf.squeeze(hessians_list))

            # Looping variables for tensorflow setup
            i = tf.constant(0)
            mc_num_tf = tf.constant(mc_num)
            #  To store final result
            l = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=1, dynamic_size=True)

            def body(i, l):
                l = l.write(i, final_stack)
                return i + 1, l

            tf_index, loop = tf.while_loop(lambda i, *_: tf.less(i, mc_num_tf), body, [i, l])

            loops = tf.cond(tf.greater(mc_num_tf, 1), lambda: tf.reduce_mean(loop.stack(), axis=0),
                            lambda: loop.stack())

            start_time = time.time()

            hessians = np.concatenate(
                [get_session().run(loops, feed_dict={input_tens: x_data[i:i + 1], keras.backend.learning_phase(): 0})
                 for i
                 in range(0, total_num)], axis=0)

            if np.all(hessians == 0.):  # warn user about not so linear activation like ReLU will get all zeros
                print(
                    'The hessians is detected to be all zeros. The common cause is you did not use any activation or '
                    'activation that is still too linear in some sense like ReLU.')

            if mean_output is True:
                hessians_master = np.mean(hessians, axis=0)
            else:
                hessians_master = np.array(hessians)

            hessians_master = np.squeeze(hessians_master)

            if denormalize:  # no need to denorm input scaling because of we assume first order dependence
                if self.labels_std is not None:
                    try:
                        hessians_master = hessians_master * self.labels_std
                    except ValueError:
                        hessians_master = hessians_master * self.labels_std.reshape(-1, 1)

            print(f'Finished hessian ({method}) calculation, {(time.time() - start_time):.{2}f} seconds elapsed')
            return hessians_master
        else:
            raise ValueError(f'Unknown method -> {method}')

    def hessian_diag(self, x=None, mean_output=False, mc_num=1, denormalize=False):
        """
        | Calculate the diagonal part of hessian of output to input, avoids the calculation of the whole hessian and takes its diagonal
        |
        | Please notice that the de-normalize (if True) assumes the output depends on the input data first orderly
        | in which the diagonal part of the hessians does not depends on input scaling and only depends on output scaling
        |
        | The diagonal part of the hessians can be all zeros and the common cause is you did not use
        | any activation or activation that is still too linear in some sense like ReLU.

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
        :History: 2018-Jun-13 - Written - Henry Leung (University of Toronto)
        """
        self.has_model_check()
        if x is None:
            raise ValueError('Please provide data to calculate the jacobian')

        if mc_num < 1 or isinstance(mc_num, float):
            raise ValueError('mc_num must be a positive integer')

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
        except ValueError:
            raise ValueError("astroNN expects input layer is named as 'input' and output layer is named as 'output', "
                             "but None is found.")

        # just in case only 1 data point is provided and mess up the shape issue
        if len(input_shape_expectation) == 3:
            x_data = np.atleast_3d(x_data)
        elif len(input_shape_expectation) == 4:
            if len(x_data.shape) < 4:
                x_data = x_data[:, :, :, np.newaxis]
        else:
            raise ValueError('Input data shape do not match neural network expectation')

        total_num = x_data.shape[0]

        hessians_diag_list = []
        for j in range(self._labels_shape):
            hessians_diag_list.append(tf.gradients(tf.gradients(output_tens[:, j], input_tens), input_tens))

        final_stack = tf.stack(tf.squeeze(hessians_diag_list))

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

        start_time = time.time()

        hessians_diag = np.concatenate(
            [get_session().run(loops, feed_dict={input_tens: x_data[i:i + 1], keras.backend.learning_phase(): 0}) for i
             in range(0, total_num)], axis=0)

        if np.all(hessians_diag == 0.):  # warn user about not so linear activation like ReLU will get all zeros
            print('The diagonal part of the hessians is detected to be all zeros. The common cause is you did not use '
                  'any activation or activation that is still too linear in some sense like ReLU.')

        if mean_output is True:
            hessians_diag_master = np.mean(hessians_diag, axis=0)
        else:
            hessians_diag_master = np.array(hessians_diag)

        hessians_diag_master = np.squeeze(hessians_diag_master)

        if denormalize:  # no need to denorm input scaling because of we assume first order dependence
            if self.labels_std is not None:
                try:
                    hessians_diag_master = hessians_diag_master * self.labels_std
                except ValueError:
                    hessians_diag_master = hessians_diag_master * self.labels_std.reshape(-1, 1)

        print(f'Finished diagonal hessian calculation, {(time.time() - start_time):.{2}f} seconds elapsed')

        return hessians_diag_master

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
            raise ValueError('Please provide data to calculate the jacobian')

        if mc_num < 1 or isinstance(mc_num, float):
            raise ValueError('mc_num must be a positive integer')

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
        except ValueError:
            raise ValueError("astroNN expects input layer is named as 'input' and output layer is named as 'output', "
                             "but None is found.")

        # just in case only 1 data point is provided and mess up the shape issue
        if len(input_shape_expectation) == 3:
            x_data = np.atleast_3d(x_data)
        elif len(input_shape_expectation) == 4:
            if len(x_data.shape) < 4:
                x_data = x_data[:, :, :, np.newaxis]
        else:
            raise ValueError('Input data shape do not match neural network expectation')

        total_num = x_data.shape[0]

        grad_list = []
        for j in range(self._labels_shape):
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
        loops = tf.reshape(loops,
                           shape=[tf.shape(input_tens)[0], *output_shape_expectation[1:], *input_shape_expectation[1:]])
        start_time = time.time()

        jacobian = np.concatenate(
            [get_session().run(loops, feed_dict={input_tens: x_data[i:i + 1], keras.backend.learning_phase(): 0}) for i
             in range(0, total_num)], axis=0)

        if mean_output is True:
            jacobian_master = np.mean(jacobian, axis=0)
        else:
            jacobian_master = np.array(jacobian)

        jacobian_master = np.squeeze(jacobian_master)

        if denormalize:
            if self.input_std is not None:
                jacobian_master = jacobian_master / np.squeeze(self.input_std)

            if self.labels_std is not None:
                try:
                    jacobian_master = jacobian_master * self.labels_std
                except ValueError:
                    jacobian_master = jacobian_master * self.labels_std.reshape(-1, 1)

        print(f'Finished all gradient calculation, {(time.time() - start_time):.{2}f} seconds elapsed')

        return jacobian_master

    @deprecated
    def jacobian_old(self, x=None, mean_output=False, denormalize=False):
        """
        | Calculate jacobian of gradient of output to input
        |
        | Please notice that the de-normalize (if True) assumes the output depends on the input data first orderly
        | in which the equation is simply jacobian divided the input scaling

        :param x: Input Data
        :type x: ndarray
        :param mean_output: False to get all jacobian, True to get the mean
        :type mean_output: boolean
        :param denormalize: De-normalize Jacobian
        :type denormalize: bool
        :History: 2017-Nov-20 - Written - Henry Leung (University of Toronto)
        """
        self.has_model_check()
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
            for j in range(self._labels_shape):
                grad_list.append(tf.gradients(output_tens[0, j], input_tens))

            final_stack = tf.stack(tf.squeeze(grad_list))
            jacobian = np.ones((x_data.shape[0], self._labels_shape, x_data.shape[1]), dtype=np.float32)

            for i in range(x_data.shape[0]):
                x_in = x_data[i:i + 1]
                jacobian[i, :, :] = get_session().run(final_stack, feed_dict={input_tens: x_in,
                                                                              keras.backend.learning_phase(): 0})

        elif len(input_shape_expectation) == 4:
            monoflag = False
            if len(x_data.shape) < 4:
                monoflag = True
                x_data = x_data[:, :, :, np.newaxis]

            jacobian = np.ones((x_data.shape[0], self._labels_shape, x_data.shape[1], x_data.shape[2], x_data.shape[3]),
                               dtype=np.float32)

            grad_list = []
            for j in range(self._labels_shape):
                grad_list.append(tf.gradients(self.keras_model.get_layer("output").output[0, j], input_tens))

            final_stack = tf.stack(tf.squeeze(grad_list))

            for i in range(x_data.shape[0]):
                x_in = x_data[i:i + 1]
                if monoflag is False:
                    jacobian[i, :, :, :, :] = get_session().run(final_stack, feed_dict={input_tens: x_in,
                                                                                        keras.backend.learning_phase(): 0})
                else:
                    jacobian[i, :, :, :, 0] = get_session().run(final_stack, feed_dict={input_tens: x_in,
                                                                                        keras.backend.learning_phase(): 0})

        else:
            raise ValueError('Input data shape do not match neural network expectation')

        if mean_output is True:
            jacobian_master = np.mean(jacobian, axis=0)
        else:
            jacobian_master = np.array(jacobian)

        jacobian_master = np.squeeze(jacobian_master)

        if denormalize:
            if self.input_std is not None:
                jacobian_master = jacobian_master / self.input_std

            if self.labels_std is not None:
                try:
                    jacobian_master = jacobian_master * self.labels_std
                except ValueError:
                    jacobian_master = jacobian_master * self.labels_std.reshape(-1, 1)

        print(f'Finished gradient calculation, {(time.time() - start_time):.{2}f} seconds elapsed')

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
            if isinstance(layer, keras.layers.Dense):
                dense_list.append(counter)

        denses = np.array(self.keras_model.layers)[dense_list]
        fig, ax = plt.subplots(1, figsize=(15, 10), dpi=100)
        for counter, dense in enumerate(denses):
            weight_temp = np.array(dense.get_weights())[0].flatten()
            ax.hist(weight_temp, 200, density=True, range=(-2., 2.), alpha=0.7,
                    label=f'Dense Layer {counter}, max: {weight_temp.max():.{2}f}, min: {weight_temp.min():.{2}f}, '
                          f'mean: {weight_temp.mean():.{2}f}, std: {weight_temp.std():.{2}f}')
        fig.suptitle(f'Dense Layers Weight Statistics of {self.folder_name}', fontsize=17)
        ax.set_xlabel('Weights', fontsize=17)
        ax.set_ylabel('Normalized Distribution', fontsize=17)
        ax.minorticks_on()
        ax.tick_params(labelsize=15, width=3, length=10, which='major')
        ax.tick_params(width=1.5, length=5, which='minor')
        ax.legend(loc='best', fontsize=15)
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
        print('==========================')
        print('This is a remainder that saving weights to h5, you might have difficult to '
              'load it back and cannot be used with astroNN probably')
        print('==========================')
        if self.fullfilepath is not None:
            return self.keras_model.save_weights(str(os.path.join(self.fullfilepath, filename)), overwrite=overwrite)
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
        return any([getattr(x, '_uses_learning_phase', False) for x in self.keras_model.outputs])

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
        keras.backend.clear_session()
