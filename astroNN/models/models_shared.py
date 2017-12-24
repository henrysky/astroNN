# ---------------------------------------------------------#
#   astroNN.models.models_shared: Shared across models
# ---------------------------------------------------------#
import os
from abc import ABCMeta, abstractmethod

import keras.backend as K
import numpy as np
from keras import regularizers
from keras.layers import MaxPooling1D, Conv1D, Dense, Dropout, Flatten
from keras.models import Model, Input
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.optimizers import Adam
from keras.backend import clear_session
from keras.models import load_model

from astroNN.shared.nn_tools import folder_runnum, cpu_fallback, gpu_memory_manage
from astroNN.NN.train_tools import threadsafe_generator
import astroNN


def load_from_folder_internal(modelobj, foldername):
    model = load_model(os.path.join(modelobj.currentdir, foldername, 'model.h5'))
    return model


class ModelStandard(object):
    """
    NAME:
        ModelStandard
    PURPOSE:
        To define astroNN standard model
    HISTORY:
        2017-Dec-23 - Written - Henry Leung (University of Toronto)
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        """
        NAME:
            model
        PURPOSE:
            To create Convolutional Neural Network model
        INPUT:
        OUTPUT:
        HISTORY:
            2017-Dec-21 - Written - Henry Leung (University of Toronto)
        """
        self.name = None
        self.__model_type = None
        self.implementation_version = None
        self.astronn_ver = astroNN.__version__
        self.batch_size = None
        self.initializer = None
        self.input_shape = None
        self.activation = None
        self.num_filters = None
        self.filter_length = None
        self.pool_length = None
        self.num_hidden = None
        self.outpot_shape = None
        self.optimizer = None
        self.currentdir = None
        self.max_epochs = None
        self.lr = None
        self.reduce_lr_epsilon = None
        self.reduce_lr_min = None
        self.reduce_lr_patience = None
        self.fallback_cpu = False
        self.limit_gpu_mem = True
        self.data_normalization = True
        self.target = None
        self.runnum_name = None
        self.fullfilepath = None

        self.beta_1 = 0.9  # exponential decay rate for the 1st moment estimates for optimization algorithm
        self.beta_2 = 0.999  # exponential decay rate for the 2nd moment estimates for optimization algorithm
        self.optimizer_epsilon = 1e-08  # a small constant for numerical stability for optimization algorithm

    def hyperparameter_writter(self):
        self.runnum_name = folder_runnum()
        self.fullfilepath = os.path.join(self.currentdir, self.runnum_name + '/')

        with open(self.fullfilepath  + 'hyperparameter_{}.txt'.format(self.runnum_name), 'w') as h:
            h.write("model: {} \n".format(self.name))
            h.write("model type: {} \n".format(self.__model_type))
            h.write("model revision version: {} \n".format(self.implementation_version))
            h.write("astroNN vesion: {} \n".format(self.astronn_ver))
            h.write("num_hidden: {} \n".format(self.num_hidden))
            h.write("num_filters: {} \n".format(self.num_filters))
            h.write("activation: {} \n".format(self.activation))
            h.write("initializer: {} \n".format(self.initializer))
            h.write("filter_length: {} \n".format(self.filter_length))
            h.write("pool_length: {} \n".format(self.pool_length))
            h.write("batch_size: {} \n".format(self.batch_size))
            h.write("max_epochs: {} \n".format(self.max_epochs))
            h.write("lr: {} \n".format(self.lr))
            h.write("reuce_lr_epsilon: {} \n".format(self.reduce_lr_epsilon))
            h.write("reduce_lr_min: {} \n".format(self.reduce_lr_min))
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

    def compile(self):
        model, linear_output, variance_output = self.model()
        model.compile(
            loss={'linear_output': self.mean_squared_error, 'variance_output': self.mse_var_wrapper([linear_output])},
            optimizer=self.optimizer, loss_weights={'linear_output': 1., 'variance_output': .2})
        return model

    def train(self, x, y):
        if self.fallback_cpu is True:
            cpu_fallback()

        if self.limit_gpu_mem is not False:
            gpu_memory_manage()

        self.hyperparameter_writter()

        self.input_shape = (x.shape[1], 1,)
        self.outpot_shape = y.shape[1]

        csv_logger = CSVLogger(self.fullfilepath + 'log.csv', append=True, separator=',')

        mean_labels = np.mean(y, axis=0)
        std_labels = np.std(y, axis=0)
        mu_std = np.vstack((mean_labels, std_labels))

        if self.optimizer is None:
            self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.optimizer_epsilon,
                                  decay=0.0)

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=self.reduce_lr_epsilon,
                                      patience=self.reduce_lr_patience, min_lr=self.reduce_lr_min, mode='min', verbose=2)
        model = self.compile()

        try:
            plot_model(model, show_shapes=True, to_file=self.fullfilepath + 'model_{}.png'.format(self.runnum_name))
        except all:
            print('Skipped plot_model! graphviz and pydot_ng are required to plot the model architecture')
            pass

        training_generator = DataGenerator(x.shape[1], self.batch_size).generate(x, y)

        model.fit_generator(generator=training_generator, steps_per_epoch=x.shape[0] // self.batch_size,
                            epochs=self.max_epochs, max_queue_size=20, verbose=2, workers=os.cpu_count(),
                            callbacks=[reduce_lr, csv_logger])

        astronn_model = 'model.h5'
        model.save(self.fullfilepath + astronn_model)
        print(astronn_model + ' saved to {}'.format(self.fullfilepath + astronn_model))
        np.save(self.fullfilepath + 'meanstd.npy', mu_std)
        np.save(self.fullfilepath + 'targetname.npy', self.target)

        clear_session()
        return model

    def load_from_folder(self, foldername):
        return load_from_folder_internal(self, foldername)

    def test(self):
        return None