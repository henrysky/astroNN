# ---------------------------------------------------------#
#   astroNN.models.models_shared: Shared across models
# ---------------------------------------------------------#
import os
import numpy as np
from abc import ABC, abstractmethod

import keras.backend as K
from keras.models import load_model
from keras.optimizers import Adam
from tensorflow.contrib import distributions
import keras
import tensorflow as tf
from tensorflow.python.client import device_lib

import astroNN
from astroNN.shared.nn_tools import folder_runnum, cpu_fallback, gpu_memory_manage


def load_from_folder_internal(modelobj, foldername):
    model = load_model(os.path.join(modelobj.currentdir, foldername, 'model.h5'))
    return model


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
        self.name = None
        self.__model_type = None
        self.implementation_version = None
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

        self.beta_1 = 0.9  # exponential decay rate for the 1st moment estimates for optimization algorithm
        self.beta_2 = 0.999  # exponential decay rate for the 2nd moment estimates for optimization algorithm
        self.optimizer_epsilon = 1e-08  # a small constant for numerical stability for optimization algorithm

    def hyperparameter_writer(self):
        self.runnum_name = folder_runnum()
        self.fullfilepath = os.path.join(self.currentdir, self.runnum_name + '/')

        with open(self.fullfilepath + 'hyperparameter.txt', 'w') as h:
            h.write("model: {} \n".format(self.name))
            h.write("astroNN internal identifier: {} \n".format(self.__model_type))
            h.write("model version: {} \n".format(self.implementation_version))
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
            h.write("neural task: {} \n".format(device_lib.list_local_devices()))
            h.write("============Tensorflow diagnostic============\n")
            h.write("\n")

            h.close()

    @abstractmethod
    def model(self):
        pass

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)

    @staticmethod
    def mse_var_wrapper(lin):
        def mse_var(y_true, y_pred):
            return K.mean(0.5 * K.square(lin - y_true) * (K.exp(-y_pred)) + 0.5 * (y_pred), axis=-1)

        return mse_var

    @staticmethod
    def categorical_cross_entropy(true, pred):
        return np.sum(true * np.log(pred), axis=1)

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

    def pre_training_checklist(self, x, y):
        if self.fallback_cpu is True:
            cpu_fallback()

        if self.limit_gpu_mem is not False:
            gpu_memory_manage()

        if self.optimizer is None or self.optimizer == 'adam':
            self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.optimizer_epsilon,
                                  decay=0.0)

        self.hyperparameter_writer()

        if self.data_normalization is True:
            mean_labels = np.median(y, axis=0)
            std_labels = np.std(y, axis=0)
            mu_std = np.vstack((mean_labels, std_labels))
            np.save(self.fullfilepath + 'meanstd.npy', mu_std)

            y = (y - mean_labels) / std_labels

        return x, y

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def train(self, x, y):
        pass

    @abstractmethod
    def test(self):
        return None
