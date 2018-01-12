# ---------------------------------------------------------#
#   astroNN.models.CVAE: Contain Variational Autoencoder Model
# ---------------------------------------------------------#
import itertools
import os

import keras.backend as K
import numpy as np
import pylab as plt
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.layers import MaxPooling1D, Conv1D, Dense, Flatten, Lambda, Reshape
from keras.models import Model, Input

from astroNN.models.CONV_VAEBase import CVAEBase
from astroNN.apogee.plotting import ASPCAP_plots
from astroNN.models.utilities.generator import VAE_DataGenerator
from astroNN.models.utilities.custom_layers import CustomVariationalLayer
from astroNN.models.utilities.normalizer import Normalizer


class APOGEE_CVAE(CVAEBase, ASPCAP_plots):
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
        super(APOGEE_CVAE, self).__init__()

        self.name = '2D Convolutional Variational Autoencoder'
        self._model_identifier = 'APOGEE_CVAE'
        self._implementation_version = '1.0'
        self.batch_size = 64
        self.initializer = 'he_normal'
        self.activation = 'relu'
        self.optimizer = 'rmsprop'
        self.num_filters = [2, 4]
        self.filter_length = 8
        self.pool_length = 4
        self.num_hidden = [128, 64]
        self.latent_dim = 2
        self.max_epochs = 100
        self.lr = 0.005
        self.reduce_lr_epsilon = 0.0005
        self.reduce_lr_min = 0.0000000001
        self.reduce_lr_patience = 4
        self.epsilon_std = 1.0
        self.task = 'regression'
        self.keras_encoder = None
        self.keras_vae = None
        self.l1 = 1e-7
        self.l2 = 1e-7

    def model(self):
        input_tensor = Input(shape=self.input_shape)
        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0],
                             kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(self.l2))(input_tensor)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[1],
                             kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(self.l2))(cnn_layer_1)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(cnn_layer_2)
        flattener = Flatten()(maxpool_1)
        layer_3 = Dense(units=self.num_hidden[0], kernel_regularizer=regularizers.l1(self.l1),
                        kernel_initializer=self.initializer, activation=self.activation)(flattener)
        layer_4 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l1(self.l1),
                        kernel_initializer=self.initializer, activation=self.activation)(layer_3)
        mean_output = Dense(units=self.latent_dim, activation="linear", name='mean_output',
                            kernel_regularizer=regularizers.l1(self.l1))(layer_4)
        sigma_output = Dense(units=self.latent_dim, activation='linear', name='sigma_output',
                             kernel_regularizer=regularizers.l1(self.l1))(layer_4)

        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([mean_output, sigma_output])

        layer_1 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l1(self.l1),
                        kernel_initializer=self.initializer, activation=self.activation)(z)
        layer_2 = Dense(units=self.num_hidden[0], kernel_regularizer=regularizers.l1(self.l1),
                        kernel_initializer=self.initializer, activation=self.activation)(layer_1)
        layer_3 = Dense(units=self.input_shape[0] * self.num_filters[1], kernel_regularizer=regularizers.l2(self.l2),
                        kernel_initializer=self.initializer, activation=self.activation)(layer_2)
        output_shape = (self.batch_size, self.input_shape[0], self.num_filters[1])
        decoder_reshape = Reshape(output_shape[1:])(layer_3)
        decnn_layer_1 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                               filters=self.num_filters[1],
                               kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(self.l2))(
            decoder_reshape)
        decnn_layer_2 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                               filters=self.num_filters[0],
                               kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(self.l2))(decnn_layer_1)
        deconv_final = Conv1D(kernel_initializer=self.initializer, activation='linear', padding="same",
                              filters=1, kernel_size=self.filter_length)(decnn_layer_2)

        y = CustomVariationalLayer()([input_tensor, deconv_final, mean_output, sigma_output])
        vae = Model(input_tensor, y)
        model_complete = Model(input_tensor, deconv_final)
        encoder = Model(input_tensor, mean_output)

        return vae, model_complete, encoder, encoder

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def train(self, input_data, input_recon_target):
        # Call the checklist to create astroNN folder and save parameters
        self.pre_training_checklist_child()

        self.input_normalizer = Normalizer(mode=self.input_norm_mode)
        self.recon_normalizer = Normalizer(mode=self.labels_norm_mode)

        norm_data, self.input_mean_norm, self.input_std_norm = self.input_normalizer.normalize(input_data)
        norm_labels, self.labels_mean_norm, self.labels_std_norm = self.recon_normalizer.normalize(input_recon_target)

        self.input_shape = (norm_data.shape[1], 1,)

        self.compile()
        self.plot_model()

        csv_logger = CSVLogger(self.fullfilepath + 'log.csv', append=True, separator=',')

        if self.task == 'classification':
            raise RuntimeError('astroNN VAE does not support classification task')

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=self.reduce_lr_epsilon,
                                      patience=self.reduce_lr_patience, min_lr=self.reduce_lr_min, mode='min',
                                      verbose=2)

        training_generator = VAE_DataGenerator(input_data.shape[1], self.batch_size).generate(input_data)

        self.keras_model.fit_generator(generator=training_generator, steps_per_epoch=input_data.shape[0] // self.batch_size,
                                       epochs=self.max_epochs, max_queue_size=20, verbose=2, workers=os.cpu_count(),
                                       callbacks=[reduce_lr, csv_logger])

        self.post_training_checklist_child()

        return None

    def test(self, input_data):
        # Prevent shallow copy issue
        input_array = np.array(input_data)
        input_array -= self.input_mean_norm
        input_array /= self.input_std_norm
        input_array = np.atleast_3d(input_array)

        print("\n")
        print('astroNN: Please ignore possible compile model warning!')
        predictions = self.keras_vae.predict(input_array)
        predictions *= self.input_std_norm
        predictions += self.input_mean_norm

        return predictions

    def test_encoder(self, input_data):
        # Prevent shallow copy issue
        input_array = np.array(input_data)
        input_array -= self.input_mean_norm
        input_array /= self.input_std_norm
        input_array = np.atleast_3d(input_array)
        return self.keras_encoder.predict(input_array)