###############################################################################
#   NeuralNetMaster.py: top-level class for a neural network
###############################################################################
import sys
from abc import ABC
import os

import keras
import keras.backend as K
from keras.utils import plot_model
import tensorflow as tf

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
            2017-Dec-23 - Written - Henry Leung (University of Toronto) \
            2018-Jan-05 - Update - Henry Leung (University of Toronto)
        """
        self.name = None
        self._model_type = None
        self._model_identifier = None
        self._implementation_version = None
        self.__python_info = sys.version
        self.__astronn_ver = astroNN.__version__
        self.__keras_ver = keras.__version__
        self.__tf_ver = tf.__version__
        self.fallback_cpu = False
        self.limit_gpu_mem = True
        self.currentdir = os.getcwd()
        self.folder_name = None
        self.fullfilepath = None

        # Hyperparameter
        self.task = None
        self.batch_size = None
        self.lr = None
        self.max_epochs = None
        self.data_normalization = None

        # optimizer parameter
        self.beta_1 = 0.9  # exponential decay rate for the 1st moment estimates for optimization algorithm
        self.beta_2 = 0.999  # exponential decay rate for the 2nd moment estimates for optimization algorithm
        self.optimizer_epsilon = K.epsilon()  # a small constant for numerical stability for optimization algorithm
        self.optimizer = None

        # Keras API
        self.keras_model = None

        self.data_normalization = None
        self.input_norm_mode = None
        self.labels_norm_mode = None

        self.input_shape = None
        self.labels_shape = None

        self.num_train = None
        self.targetname = None

    def pre_training_checklist_master(self):
        if self.fallback_cpu is True:
            cpu_fallback()

        if self.limit_gpu_mem is False:
            gpu_memory_manage()
        elif isinstance(self.limit_gpu_mem, float) is True:
            gpu_memory_manage(ratio=self.limit_gpu_mem)

        if self.data_normalization is False:
            self.input_norm_mode = 0
            self.labels_norm_mode = 0

        self.folder_name = folder_runnum()
        self.fullfilepath = os.path.join(self.currentdir, self.folder_name + '/')

        with open(self.fullfilepath + 'hyperparameter.txt', 'w') as h:
            h.write("model: {} \n".format(self.name))
            h.write("astroNN internal identifier: {} \n".format(self._model_type))
            h.write("model version: {} \n".format(self.__python_info))
            h.write("astroNN version: {} \n".format(self.__astronn_ver))
            h.write("keras version: {} \n".format(self.__keras_ver))
            h.write("tensorflow version: {} \n".format(self.__tf_ver))
            h.write("folder name: {} \n".format(self.folder_name))
            h.write("fallback cpu? : {} \n".format(self.fallback_cpu))
            h.write("astroNN GPU management: {} \n".format(self.limit_gpu_mem))
            h.write("batch_size: {} \n".format(self.batch_size))
            h.write("optimizer: {} \n".format(self.optimizer))
            h.write("max_epochs: {} \n".format(self.max_epochs))
            h.write("learning rate: {} \n".format(self.lr))
            h.close()

    def post_training_checklist_master(self):
        pass

    def plot_model(self):
        try:
            plot_model(self.keras_model, show_shapes=True, to_file=self.fullfilepath + 'model.png')
        except all:
            print('Skipped plot_model! graphviz and pydot_ng are required to plot the model architecture')
            pass
