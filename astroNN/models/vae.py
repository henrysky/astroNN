# ---------------------------------------------------------#
#   astroNN.models.vae: Contain Variational Autoencoder Model
# ---------------------------------------------------------#
import itertools
import os

import keras.backend as K
import numpy as np
import pylab as plt
from keras import metrics
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.layers import MaxPooling1D, Conv1D, Dense, Flatten, Lambda, Layer, Reshape
from keras.models import Model, Input
from keras.models import load_model

from astroNN.models import ModelStandard
from astroNN.models.models_tools import threadsafe_generator


class VAE(ModelStandard):
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
        super(VAE, self).__init__()

        self.name = 'Convolutional Variational Autoencoder'
        self._model_type = 'CVAE'
        self._implementation_version = '1.0'
        self.batch_size = 64
        self.initializer = 'he_normal'
        self.activation = 'relu'
        self.num_filters = [2, 4]
        self.filter_length = 8
        self.pool_length = 4
        self.num_hidden = [196, 96]
        self.latent_dim = 2
        self.max_epochs = 40
        self.lr = 0.005
        self.reduce_lr_epsilon = 0.00005
        self.reduce_lr_min = 0.0000000001
        self.reduce_lr_patience = 10
        self.epsilon_std = 1.0
        self.data_normalization = False
        self.task = 'regression'

    def model(self):
        input_tensor = Input(shape=self.input_shape)
        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0],
                             kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(1e-4))(input_tensor)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[1],
                             kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(1e-4))(cnn_layer_1)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(cnn_layer_2)
        flattener = Flatten()(maxpool_1)
        layer_3 = Dense(units=self.num_hidden[0], kernel_regularizer=regularizers.l2(1e-4),
                        kernel_initializer=self.initializer, activation=self.activation)(flattener)
        layer_4 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l2(1e-4),
                        kernel_initializer=self.initializer, activation=self.activation)(layer_3)
        mean_output = Dense(units=self.latent_dim, activation="linear", name='mean_output')(layer_4)
        sigma_output = Dense(units=self.latent_dim, activation='linear', name='sigma_output')(layer_4)

        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([mean_output, sigma_output])

        layer_1 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l2(1e-4),
                        kernel_initializer=self.initializer, activation=self.activation)(z)
        layer_2 = Dense(units=self.num_hidden[0], kernel_regularizer=regularizers.l2(1e-4),
                        kernel_initializer=self.initializer, activation=self.activation)(layer_1)
        layer_3 = Dense(units=self.input_shape[0] * self.num_filters[1], kernel_regularizer=regularizers.l2(1e-4),
                        kernel_initializer=self.initializer, activation=self.activation)(layer_2)
        output_shape = (self.batch_size, self.input_shape[0], self.num_filters[1])
        decoder_reshape = Reshape(output_shape[1:])(layer_3)
        decnn_layer_1 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                               filters=self.num_filters[1],
                               kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(1e-4))(
            decoder_reshape)
        decnn_layer_2 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                               filters=self.num_filters[0],
                               kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(1e-4))(decnn_layer_1)
        deconv_final = Conv1D(kernel_initializer=self.initializer, activation='linear', padding="same",
                              filters=1, kernel_size=self.filter_length)(decnn_layer_2)

        y = CustomVariationalLayer()([input_tensor, deconv_final, mean_output, sigma_output])
        vae = Model(input_tensor, y)
        model_complete = Model(input_tensor, deconv_final)
        encoder = Model(input_tensor, mean_output)

        return vae, encoder, model_complete

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def compile(self):
        model, encoder, model_test = self.model()
        model.compile(loss=None, optimizer=self.optimizer)
        return model, encoder, model_test

    def train(self, x, y):
        x, y = super().train(x, y)

        csv_logger = CSVLogger(self.fullfilepath + 'log.csv', append=True, separator=',')

        if self.task == 'classification':
            raise RuntimeError('astroNN VAE does not support classification task')

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=self.reduce_lr_epsilon,
                                      patience=self.reduce_lr_patience, min_lr=self.reduce_lr_min, mode='min',
                                      verbose=2)
        model, encoder, model_complete = self.compile()

        self.plot_model(model)

        training_generator = DataGenerator(x.shape[1], self.batch_size).generate(x)

        model.fit_generator(generator=training_generator, steps_per_epoch=x.shape[0] // self.batch_size,
                            epochs=self.max_epochs, max_queue_size=20, verbose=2, workers=os.cpu_count(),
                            callbacks=[reduce_lr, csv_logger])

        astronn_model = 'model.h5'
        astronn_encoder = 'encoder.h5'
        model_complete.save(self.fullfilepath + astronn_model)
        encoder.save(self.fullfilepath + astronn_encoder)
        print(astronn_model + ' saved to {}'.format(self.fullfilepath + astronn_model))
        print(astronn_model + ' saved to {}'.format(self.fullfilepath + astronn_encoder))

        return model, encoder, model_complete

    @staticmethod
    def plot_latent(pred):
        N = pred.shape[1]
        for i, j in itertools.product(range(N), range(N)):
            if i != j and j > i:
                plt.figure(figsize=(15,11), dpi=200)
                plt.scatter(pred[:, i], pred[:, j], s=0.9)
                plt.title('Latent Variable {} against {}'.format(i, j))
                plt.xlabel('Latent Variable {}'.format(i))
                plt.ylabel('Latent Variable {}'.format(j))
        plt.show()

    def test(self, x):
        x = super().test(x)
        model = load_model(self.fullfilepath + 'model_complete.h5', custom_objects={'CustomVariationalLayer': CustomVariationalLayer})
        print("\n")
        print('astroNN: Please ignore possible compile model warning!')
        return model.predict(x)

    def test_encoder(self, x):
        x, model = super().test(x)
        encoder = load_model(self.fullfilepath + 'encoder.h5', custom_objects={'CustomVariationalLayer': CustomVariationalLayer})
        print('astroNN: Please ignore possible compile model warning!')
        return encoder.predict(x)


class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    @staticmethod
    def vae_loss(x, x_decoded_mean, z_mean, z_log_var):
        shape = int(x.shape[1])
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = shape * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        z_mean = inputs[2]
        z_log_var = inputs[3]
        loss = self.vae_loss(x, x_decoded_mean, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x


class DataGenerator(object):
    """
    NAME:
        DataGenerator
    PURPOSE:
        To generate data for Keras
    INPUT:
    OUTPUT:
    HISTORY:
        2017-Dec-02 - Written - Henry Leung (University of Toronto)
        2017-Dec-21 - Update - Henry Leung (University of Toronto)
    """

    def __init__(self, dim, batch_size, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __get_exploration_order(self, list_IDs):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle is True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, spectra, list_IDs_temp):
        'Generates data of batch_size samples'
        # X : (n_samples, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim, 1))

        # Generate data
        X[:, :, 0] = spectra[list_IDs_temp]

        return X

    @threadsafe_generator
    def generate(self, input):
        'Generates batches of samples'
        # Infinite loop
        list_IDs = range(input.shape[0])
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)

            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = indexes[i * self.batch_size:(i + 1) * self.batch_size]

                # Generate data
                X = self.__data_generation(input, list_IDs_temp)

                yield X, None
