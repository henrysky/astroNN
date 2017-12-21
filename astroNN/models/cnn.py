# ---------------------------------------------------------#
#   astroNN.models.cnn: Contain CNN Model
# ---------------------------------------------------------#
import random

import keras.backend as K
import numpy as np
from keras import regularizers
from keras.layers import MaxPooling1D, Conv1D, Dense, Dropout
from keras.models import Model, Input


class CNN(object):
    def __init__(self):
        self.batch_size = 64
        self.initializer = 'he_normal'
        self.input_shape = None
        self.activation = 'relu'
        self.num_filters = [2, 4]
        self.filter_length = 8
        self.pool_length = 4
        self.num_hidden = [196, 96]
        self.num_labels = None
        self.optimizer = 'adam'
        print("CNN")

    def model(self):
        """
        NAME:
            model
        PURPOSE:
            To create Convolutional Neural Network model 1 for apogee
        INPUT:
        OUTPUT:
        HISTORY:
            2017-Oct-14 Henry Leung
        """
        input_tensor = Input(batch_shape=self.input_shape)
        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0],
                             kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(1e-4))(input_tensor)
        dropout_1 = Dropout(0.3)(cnn_layer_1)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0],
                             kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(1e-4))(dropout_1)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(cnn_layer_2)
        dropout_2 = Dropout(0.3)(maxpool_1)
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
        return
