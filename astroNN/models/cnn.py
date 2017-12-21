# ---------------------------------------------------------#
#   astroNN.models.cnn: Contain CNN Model
# ---------------------------------------------------------#
import os
import random

import keras.backend as K
import numpy as np
from keras import regularizers
from keras.layers import MaxPooling1D, Conv1D, Dense, Dropout, Flatten
from keras.models import Model, Input
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.optimizers import Adam

from astroNN.shared.nn_tools import folder_runnum, cpu_fallback, gpu_memory_manage


class CNN(object):
    """
    NAME:
        CNN
    PURPOSE:
        To create Convolutional Neural Network model
    HISTORY:
        2017-Dec-21 - Written - Henry Leung (University of Toronto)
    """

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
        self.name = 'CNN'
        self.batch_size = 64
        self.initializer = 'he_normal'
        self.input_shape = None
        self.activation = 'relu'
        self.num_filters = [2, 4]
        self.filter_length = 8
        self.pool_length = 4
        self.num_hidden = [196, 96]
        self.num_labels = None
        self.optimizer = None
        self.currentdir = os.getcwd()
        self.max_epochs = 500
        self.lr = 0.005
        self.reuce_lr_epsilon = 0.00005
        self.reduce_lr_min = 0.0000000001
        self.reduce_lr_patience = 10
        self.fallback_cpu = False
        self.limit_gpu_mem = True
        self.target = 'all'

        self.beta_1 = 0.9  # exponential decay rate for the 1st moment estimates for optimization algorithm
        self.beta_2 = 0.999  # exponential decay rate for the 2nd moment estimates for optimization algorithm
        self.optimizer_epsilon = 1e-08  # a small constant for numerical stability for optimization algorithm

    def hyperparameter_writter(self):
        self.runnum_name = folder_runnum()
        self.fullfilepath = os.path.join(self.currentdir, self.runnum_name + '/')

        with open(self.fullfilepath  + 'hyperparameter_{}.txt'.format(self.fullfilepath), 'w') as h:
            h.write("model: {} \n".format(self.name))
            h.write("num_hidden: {} \n".format(self.num_hidden))
            h.write("num_filters: {} \n".format(self.num_filters))
            h.write("activation: {} \n".format(self.activation))
            h.write("initializer: {} \n".format(self.initializer))
            h.write("filter_length: {} \n".format(self.filter_length))
            h.write("pool_length: {} \n".format(self.pool_length))
            h.write("batch_size: {} \n".format(self.batch_size))
            h.write("max_epochs: {} \n".format(self.max_epochs))
            h.write("lr: {} \n".format(self.lr))
            h.write("reuce_lr_epsilon: {} \n".format(self.reuce_lr_epsilon))
            h.write("reduce_lr_min: {} \n".format(self.reduce_lr_min))
            h.close()

    def model(self):
        input_tensor = Input(batch_shape=self.input_shape)
        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0],
                             kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(1e-4))(input_tensor)
        dropout_1 = Dropout(0.3)(cnn_layer_1)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0],
                             kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(1e-4))(dropout_1)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(cnn_layer_2)
        flattener = Flatten()(maxpool_1)
        dropout_2 = Dropout(0.3)(flattener)
        layer_3 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l2(1e-4),
                        kernel_initializer=self.initializer,
                        activation=self.activation)(dropout_2)
        dropout_3 = Dropout(0.3)(layer_3)
        layer_4 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l2(1e-4),
                        kernel_initializer=self.initializer,
                        activation=self.activation)(dropout_3)
        linear_output = Dense(units=self.num_labels, activation="linear", name='linear_output')(layer_4)
        variance_output = Dense(units=self.num_labels, activation='linear', name='variance_output')(layer_4)

        model = Model(inputs=input_tensor, outputs=[variance_output, linear_output])

        return model, linear_output, variance_output

    def mean_squared_error(self, y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)

    def mse_var_wrapper(self, lin):
        def mse_var(y_true, y_pred):
            return K.mean(0.5 * K.square(lin - y_true) * (K.exp(-y_pred)) + 0.5 * (y_pred), axis=-1)

        return mse_var

    def generate_train_batch(self, x, y):
        while True:
            indices = random.sample(range(0, x.shape[0]), self.batch_size)
            indices = np.sort(indices)
            x_batch, y_batch = x[indices], y[indices]
            yield (x_batch, {'linear_output': y_batch, 'variance_output': y_batch})

    def compile(self):
        model, linear_output, variance_output = self.model()
        model.compile(
            loss={'linear_output': self.mean_squared_error, 'variance_output': self.mse_var_wrapper([linear_output])},
            optimizer=self.optimizer, loss_weights={'linear_output': 1., 'variance_output': .2})
        return model

    def train(self):
        if self.fallback_cpu is True:
            cpu_fallback()

        if self.limit_gpu_mem is not False:
            gpu_memory_manage()

        self.hyperparameter_writter()

        csv_logger = CSVLogger(self.fullfilepath + 'log.csv', append=True, separator=',')

        if self.optimizer is None:
            self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.optimizer_epsilon,
                                  decay=0.0)

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=self.reuce_lr_epsilon,
                                      patience=self.reduce_lr_patience, min_lr=self.reduce_lr_min, mode='min', verbose=2)
        model = self.compile()

        try:
            plot_model(model, show_shapes=True, to_file=self.fullfilepath + 'model_{}.png'.format(self.runnum_name))
        except:
            pass


        return