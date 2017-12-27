# ---------------------------------------------------------#
#   astroNN.models.models_shared: Shared across models
# ---------------------------------------------------------#
import os
from abc import ABC, abstractmethod

import keras.backend as K
import numpy as np
from keras.models import load_model

from astroNN.shared.nn_tools import folder_runnum, cpu_fallback, gpu_memory_manage
import astroNN


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
        self.astronn_ver = astroNN.__version__
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

    def hyperparameter_writter(self):
        self.runnum_name = folder_runnum()
        self.fullfilepath = os.path.join(self.currentdir, self.runnum_name + '/')

        with open(self.fullfilepath + 'hyperparameter.txt', 'w') as h:
            h.write("model: {} \n".format(self.name))
            h.write("astroNN internal identifier: {} \n".format(self.__model_type))
            h.write("model version: {} \n".format(self.implementation_version))
            h.write("astroNN version: {} \n".format(self.astronn_ver))
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

            h.close()

    @abstractmethod
    def model(self):
        pass

    def mean_squared_error(self, y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)

    def mse_var_wrapper(self, lin):
        def mse_var(y_true, y_pred):
            return K.mean(0.5 * K.square(lin - y_true) * (K.exp(-y_pred)) + 0.5 * (y_pred), axis=-1)
        return mse_var

    def pre_training_checklist(self):
        if self.fallback_cpu is True:
            cpu_fallback()

        if self.limit_gpu_mem is not False:
            gpu_memory_manage()

        self.hyperparameter_writter()

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def test(self):
        return None
