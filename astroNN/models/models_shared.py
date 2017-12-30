# ---------------------------------------------------------#
#   astroNN.models.models_shared: Shared across models
# ---------------------------------------------------------#
import os
import sys
from abc import ABC, abstractmethod

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.utils import plot_model
from tensorflow.contrib import distributions
from tensorflow.python.client import device_lib
import keras.losses

import astroNN
from astroNN.shared.nn_tools import folder_runnum, cpu_fallback, gpu_memory_manage

K.set_learning_phase(1)


class ModelStandard(ABC):
    """
    NAME:
        ModelStandard
    PURPOSE:
        To define astroNN standard model
    HISTORY:
        2017-Dec-23 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self):
        keras.losses.custom_loss = self.mse_var_wrapper

        self.name = None
        self._model_type = None
        self._implementation_version = None
        self.__python_info = sys.version
        self.__astronn_ver = astroNN.__version__
        self.__keras_ver = keras.__version__
        self.__tf_ver = tf.__version__
        self.runnum_name = None
        self.batch_size = None
        self.initializer = None
        self.input_shape = None
        self.activation = None
        self.num_filters = 'N/A'
        self.filter_length = 'N/A'
        self.pool_length = 'N/A'
        self.num_hidden = None
        self.output_shape = None
        self.optimizer = None
        self.max_epochs = None
        self.latent_dim = 'N/A'
        self.lr = None
        self.reduce_lr_epsilon = None
        self.reduce_lr_min = None
        self.reduce_lr_patience = None
        self.fallback_cpu = False
        self.limit_gpu_mem = True
        self.data_normalization = True
        self.target = None
        self.currentdir = os.getcwd()
        self.fullfilepath = None
        self.task = 'regression'  # Either 'regression' or 'classification'
        self.keras_model = None

        self.beta_1 = 0.9  # exponential decay rate for the 1st moment estimates for optimization algorithm
        self.beta_2 = 0.999  # exponential decay rate for the 2nd moment estimates for optimization algorithm
        self.optimizer_epsilon = 1e-08  # a small constant for numerical stability for optimization algorithm

    def hyperparameter_writer(self):
        with open(self.fullfilepath + 'hyperparameter.txt', 'w') as h:
            h.write("model: {} \n".format(self.name))
            h.write("astroNN internal identifier: {} \n".format(self._model_type))
            h.write("model version: {} \n".format(self._implementation_version))
            h.write("python version: {} \n".format(self.__python_info))
            h.write("astroNN version: {} \n".format(self.__astronn_ver))
            h.write("keras version: {} \n".format(self.__keras_ver))
            h.write("tensorflow version: {} \n".format(self.__tf_ver))
            h.write("runnum_name: {} \n".format(self.runnum_name))
            h.write("batch_size: {} \n".format(self.batch_size))
            h.write("initializer: {} \n".format(self.initializer))
            h.write("input_shape: {} \n".format(self.input_shape))
            h.write("activation: {} \n".format(self.activation))
            h.write("num_filters: {} \n".format(self.num_filters))
            h.write("filter_length: {} \n".format(self.filter_length))
            h.write("pool_length: {} \n".format(self.pool_length))
            h.write("num_hidden: {} \n".format(self.num_hidden))
            h.write("output_shape: {} \n".format(self.output_shape))
            h.write("optimizer: {} \n".format(self.optimizer))
            h.write("max_epochs: {} \n".format(self.max_epochs))
            h.write("latent dimension: {} \n".format(self.latent_dim))
            h.write("lr: {} \n".format(self.lr))
            h.write("reduce_lr_epsilon: {} \n".format(self.reduce_lr_epsilon))
            h.write("reduce_lr_min: {} \n".format(self.reduce_lr_min))
            h.write("reduce_lr_patience: {} \n".format(self.reduce_lr_patience))
            h.write("fallback cpu? : {} \n".format(self.fallback_cpu))
            h.write("astroNN GPU management: {} \n".format(self.limit_gpu_mem))
            h.write("astroNN data normalizing implementation? : {} \n".format(self.data_normalization))
            h.write("target? : {} \n".format(self.target))
            h.write("currentdir: {} \n".format(self.currentdir))
            h.write("fullfilepath: {} \n".format(self.fullfilepath))
            h.write("neural task: {} \n".format(self.task))
            h.write("\n")
            h.write("============Tensorflow diagnostic============\n")
            h.write("{} \n".format(device_lib.list_local_devices()))
            h.write("============Tensorflow diagnostic============\n")
            h.write("\n")

            h.close()

            astronn_internal_path = os.path.join(self.fullfilepath, 'astroNN_use_only')
            os.makedirs(astronn_internal_path)

            np.save(astronn_internal_path + '/astroNN_identifier.npy', self._model_type)
            np.save(astronn_internal_path + '/input.npy', self.input_shape)
            np.save(astronn_internal_path + '/output.npy', self.output_shape)

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return K.mean(K.tf.where(K.tf.equal(y_true, -9999.), K.tf.zeros_like(y_true), K.square(y_true - y_pred)), axis=-1)

    @staticmethod
    def mse_var_wrapper(lin):
        def mse_var(y_true, y_pred):
            lin2 = K.tf.reshape(lin, K.tf.shape(y_pred))
            return K.mean(K.tf.where(K.tf.equal(y_true, -9999.), K.tf.zeros_like(y_true),
                                     0.5*K.square(y_true-lin2)*(K.exp(-y_pred)) + 0.5*y_pred), axis=-1)
            # return K.mean(K.switch(K.equal(y_true, -9999.), K.tf.zeros_like(y_true),
            #                        0.5 * K.square(K.tf.squeeze(lin) - y_true) * (K.exp(-y_pred)) + 0.5 * y_pred), axis=-1)
        return mse_var

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        return K.sum(K.switch(K.equal(y_true, -9999.), K.tf.zeros_like(y_true), y_true * np.log(y_pred)), axis=-1)

    @staticmethod
    def gaussian_categorical_crossentropy(true, pred, dist, undistorted_loss, num_classes):
        # for a single monte carlo simulation,
        #   calculate categorical_crossentropy of
        #   predicted logit values plus gaussian
        #   noise vs true values.
        # true - true values. Shape: (N, C)
        # pred - predicted logit values. Shape: (N, C)
        # dist - normal distribution to sample from. Shape: (N, C)
        # undistorted_loss - the crossentropy loss without variance distortion. Shape: (N,)
        # num_classes - the number of classes. C
        # returns - total differences for all classes (N,)
        def map_fn(i):
            std_samples = K.transpose(dist.sample(num_classes))
            distorted_loss = K.categorical_crossentropy(pred + std_samples, true, from_logits=True)
            diff = undistorted_loss - distorted_loss
            return -K.elu(diff)

        return map_fn

    def bayesian_categorical_crossentropy(self, T, num_classes):
        # Bayesian categorical cross entropy.
        # N data points, C classes, T monte carlo simulations
        # true - true values. Shape: (N, C)
        # pred_var - predicted logit values and variance. Shape: (N, C + 1)
        # returns - loss (N,)
        def bayesian_categorical_crossentropy_internal(true, pred_var):
            # shape: (N,)
            std = K.sqrt(pred_var[:, num_classes:])
            # shape: (N,)
            variance = pred_var[:, num_classes]
            variance_depressor = K.exp(variance) - K.ones_like(variance)
            # shape: (N, C)
            pred = pred_var[:, 0:num_classes]
            # shape: (N,)
            undistorted_loss = K.categorical_crossentropy(pred, true, from_logits=True)
            # shape: (T,)
            iterable = K.variable(np.ones(T))
            dist = distributions.Normal(loc=K.zeros_like(std), scale=std)
            monte_carlo_results = K.map_fn(
                self.gaussian_categorical_crossentropy(true, pred, dist, undistorted_loss, num_classes), iterable,
                name='monte_carlo_results')

            variance_loss = K.mean(monte_carlo_results, axis=0) * undistorted_loss

            return variance_loss + undistorted_loss + variance_depressor

        return bayesian_categorical_crossentropy_internal

    def pre_training_checklist(self, x_data, y_data):
        if self.fallback_cpu is True:
            cpu_fallback()

        if self.limit_gpu_mem is not False:
            gpu_memory_manage()

        if self.task != 'regression' and self.task != 'classification':
            raise RuntimeError("task can only either be 'regression' or 'classification'. ")

        self.runnum_name = folder_runnum()
        self.fullfilepath = os.path.join(self.currentdir, self.runnum_name + '/')

        if self.optimizer is None or self.optimizer == 'adam':
            self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.optimizer_epsilon,
                                  decay=0.0)

        if self.data_normalization is True:
            # do not include -9999 in mean and std calculation and do not normalize those elements because
            # astroNN is designed to ignore -9999. only
            mean_labels = np.zeros(y_data.shape[1])
            std_labels = np.ones(y_data.shape[1])
            for i in range(y_data.shape[1]):
                not9999 = np.where(y_data[:, i] != -9999.)[0]
                mean_labels[i] = np.median((y_data[:, i])[not9999], axis=0)
                std_labels[i] = np.std((y_data[:, i])[not9999], axis=0)
                (y_data[:, i])[not9999] -= mean_labels[i]
                (y_data[:, i])[not9999] /= std_labels[i]
            mu_std = np.vstack((mean_labels, std_labels))
            np.save(self.fullfilepath + 'meanstd.npy', mu_std)
            np.save(self.fullfilepath + 'targetname.npy', self.target)

            x_mu_std = np.vstack((np.median(x_data), np.std(x_data)))
            np.save(self.fullfilepath + 'meanstd_x.npy', x_mu_std)

            x_data -= x_mu_std[0]
            x_data /= x_mu_std[1]

        self.input_shape = (x_data.shape[1], 1,)
        self.output_shape = (y_data.shape[1], 1,)

        self.hyperparameter_writer()

        return x_data, y_data

    def model_existence(self):
        if self.keras_model is None:
            try:
                self.keras_model.load_weights(self.fullfilepath + '/model_weights.h5')
            except all:
                raise TypeError('This object contains no model, Please load the model first')

    def plot_model(self):
        try:
            plot_model(self.keras_model, show_shapes=True, to_file=self.fullfilepath + 'model.png')
        except all:
            print('Skipped plot_model! graphviz and pydot_ng are required to plot the model architecture')
            pass

    def jacobian(self, x=None):
        """
        NAME: cal_jacobian
        PURPOSE: calculate jacobian
        INPUT:
        OUTPUT:
        HISTORY:
            2017-Nov-20 Henry Leung
        """
        if x is None:
            raise ValueError('Please provide data to calculate the jacobian')

        K.set_learning_phase(0)

        # Force to reload model to start a new session
        self.model_existence()
        x = np.atleast_3d(x)
        # enforce float16 because the precision doesnt really matter
        input_tens = self.keras_model.layers[0].input
        jacobian = np.empty((self.output_shape[0], x.shape[0], x.shape[1]), dtype=np.float16)
        with K.get_session() as sess:
            for j in range(self.output_shape[0]):
                grad = self.keras_model.layers[-1].output[0, j]
                for i in range(x.shape[0]):
                    jacobian[j, i:i + 1, :] = (np.asarray(sess.run(K.tf.gradients(grad, input_tens),
                                                                   feed_dict={input_tens: x[i:i + 1]})))[:, :, 0].T
        return jacobian

    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def train(self, x, y):
        x, y = self.pre_training_checklist(x, y)
        return x, y

    @abstractmethod
    def test(self, x):
        mustd_x = np.load(self.fullfilepath + '/meanstd_x.npy')
        x -= mustd_x[0]
        x /= mustd_x[1]
        x = np.atleast_3d(x)
        self.model_existence()
        return x

    def aspcap_residue_plot(self, test_predictions, test_labels, test_pred_error):
        import pylab as plt
        from astroNN.shared.nn_tools import target_name_conversion
        import numpy as np
        from astropy.stats import mad_std
        import seaborn as sns

        resid = test_predictions - test_labels

        # Some plotting variables for asthetics
        plt.rcParams['axes.facecolor'] = 'white'
        sns.set_style("ticks")
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.color'] = 'gray'
        plt.rcParams['grid.alpha'] = '0.4'
        std_labels = np.load(self.fullfilepath + '/meanstd.npy')[1]

        x_lab = 'ASPCAP'
        y_lab = 'astroNN'
        fullname = target_conversion(self.target)

        for i in range(self.output_shape[0]):
            plt.figure(figsize=(15, 11), dpi=200)
            plt.axhline(0, ls='--', c='k', lw=2)
            not9999 = np.where(test_labels[:, i] != -9999.)[0]
            plt.errorbar((test_labels[:, i])[not9999], (resid[:, i])[not9999], yerr=(test_pred_error[:,i])[not9999],
                         markersize=2, fmt='o', ecolor='g', capthick=2, elinewidth=0.5)

            plt.xlabel('ASPCAP ' + target_name_conversion(fullname[i]), fontsize=25)
            plt.ylabel('$\Delta$ ' + target_name_conversion(fullname[i]) + '\n(' + y_lab + ' - ' + x_lab + ')', fontsize=25)
            plt.tick_params(labelsize=20, width=1, length=10)
            if self.output_shape[0] == 1:
                plt.xlim([np.min((test_labels[:])[not9999]), np.max((test_labels[:])[not9999])])
            else:
                plt.xlim([np.min((test_labels[:, i])[not9999]), np.max((test_labels[:, i])[not9999])])
            ranges = (np.max((test_labels[:, i])[not9999]) - np.min((test_labels[:, i])[not9999])) / 2
            plt.ylim([-ranges, ranges])
            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=2)
            bias = np.median((resid[:, i])[not9999])
            scatter = np.std((resid[:, i])[not9999])
            plt.figtext(0.6, 0.75,
                        '$\widetilde{m}$=' + '{0:.3f}'.format(bias) + ' $\widetilde{s}$=' + '{0:.3f}'.format(
                            scatter / float(std_labels[i])) + ' s=' + '{0:.3f}'.format(scatter), size=25, bbox=bbox_props)
            plt.tight_layout()
            plt.savefig(self.fullfilepath + '/{}_test.png'.format(fullname[i]))
            plt.close('all')
            plt.clf()


def target_conversion(target):
    if target == 'all' or target == ['all']:
        target = ['teff', 'logg', 'M', 'alpha', 'C', 'C1', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Ca', 'Ti',
                  'Ti2', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'absmag']
    return np.asarray(target)
