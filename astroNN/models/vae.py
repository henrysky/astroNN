# ---------------------------------------------------------#
#   astroNN.models.vae: Contain Variational Autoencoder Model
# ---------------------------------------------------------#
import numpy as np
import random

import keras.backend as K
from keras import regularizers
from keras.layers import MaxPooling1D, UpSampling1D, Conv1D, Dense, Dropout, Flatten, Lambda, Layer
from keras.models import Model, Input


class VAE(object):
    """
    NAME:
        VAE
    PURPOSE:
        To create Variational Autoencoder
    HISTORY:
        2017-Dec-21 - Written - Henry Leung (University of Toronto)
    """
    def __init__(self):
        """
        NAME:
            model
        PURPOSE:
            To create Variational Autoencoder
        INPUT:
        OUTPUT:
        HISTORY:
            2017-Dec-21 - Written - Henry Leung (University of Toronto)
        """
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
        self.latent_size = 2
        self.epsilon_std = 1.0

    def model(self):
        input_tensor = Input(batch_shape=self.input_shape)
        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0],
                             kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(1e-4))(input_tensor)
        # dropout_1 = Dropout(0.3)(cnn_layer_1)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0],
                             kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(1e-4))(cnn_layer_1)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(cnn_layer_2)
        # dropout_2 = Dropout(0.3)(maxpool_1)
        flattener = Flatten()(maxpool_1)
        layer_3 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l2(1e-4),
                        kernel_initializer=self.initializer,
                        activation=self.activation)(flattener)
        # dropout_3 = Dropout(0.3)(layer_3)
        layer_4 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l2(1e-4),
                        kernel_initializer=self.initializer,
                        activation=self.activation)(layer_3)
        mean_output = Dense(units=self.num_labels, activation="linear", name='mean_output')(layer_4)
        sigma_output = Dense(units=self.num_labels, activation='linear', name='sigma_output')(layer_4)

        z = Lambda(self.sampling, output_shape=(self.latent_size,))([mean_output, sigma_output])

        layer_1 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l2(1e-4),
                        kernel_initializer=self.initializer,
                        activation=self.activation)(z)
        layer_2 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l2(1e-4),
                        kernel_initializer=self.initializer,
                        activation=self.activation)(layer_1)
        upsample_1 = UpSampling1D(pool_size=self.pool_length)(layer_2)
        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[1],
                             kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(1e-4))(upsample_1)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0],
                             kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(1e-4))(cnn_layer_1)
        flattener = Flatten()(cnn_layer_2)

        output = Dense(units=self.num_labels, activation="linear", name='output')(flattener)

        encoder_model = Model(inputs=input_tensor, outputs=[mean_output])
        decoder_model = Model(inputs=mean_output, outputs=output)
        model = Model(inputs=input_tensor, outputs=[mean_output, sigma_output])

        return model, encoder_model, decoder_model

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_size), mean=0., stddev=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

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
