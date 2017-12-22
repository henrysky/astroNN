# ---------------------------------------------------------#
#   astroNN.models.vae: Contain Variational Autoencoder Model
# ---------------------------------------------------------#
import numpy as np
import os

from astroNN.NN.train_tools import threadsafe_generator
from astroNN.shared.nn_tools import folder_runnum, cpu_fallback, gpu_memory_manage

import keras.backend as K
from keras import regularizers
from keras.layers import MaxPooling1D, UpSampling1D, Conv1D, Dense, Dropout, Flatten, Lambda, Layer, Reshape
from keras.models import Model, Input
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.optimizers import Adam
from keras.backend import set_session, clear_session
from keras import metrics


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
        self.name = 'Vational Autoencoder'
        self.batch_size = 64
        self.initializer = 'he_normal'
        self.input_shape = None
        self.activation = 'relu'
        self.num_filters = [2, 4]
        self.filter_length = 8
        self.pool_length = 4
        self.num_hidden = [196, 96]
        self.outpot_shape = None
        self.optimizer = 'rmsprop'
        self.latent_size = 2
        self.max_epochs = 500
        self.lr = 0.005
        self.reduce_lr_epsilon = 0.00005
        self.reduce_lr_min = 0.0000000001
        self.reduce_lr_patience = 10
        self.epsilon_std = 1.0
        self.fallback_cpu = False
        self.limit_gpu_mem = True
        self.currentdir = os.getcwd()

        self.beta_1 = 0.9  # exponential decay rate for the 1st moment estimates for optimization algorithm
        self.beta_2 = 0.999  # exponential decay rate for the 2nd moment estimates for optimization algorithm
        self.optimizer_epsilon = 1e-08  # a small constant for numerical stability for optimization algorithm

    def hyperparameter_writter(self):
        self.runnum_name = folder_runnum()
        self.fullfilepath = os.path.join(self.currentdir, self.runnum_name + '/')

        with open(self.fullfilepath + 'hyperparameter_{}.txt'.format(self.runnum_name), 'w') as h:
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
            h.write("reuce_lr_epsilon: {} \n".format(self.reduce_lr_epsilon))
            h.write("reduce_lr_min: {} \n".format(self.reduce_lr_min))
            h.close()

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
        mean_output = Dense(units=self.latent_size, activation="linear", name='mean_output')(layer_4)
        sigma_output = Dense(units=self.latent_size, activation='linear', name='sigma_output')(layer_4)

        z = Lambda(self.sampling, output_shape=(self.latent_size,))([mean_output, sigma_output])

        layer_1 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l2(1e-4),
                        kernel_initializer=self.initializer, activation=self.activation)(z)
        layer_2 = Dense(units=self.num_hidden[0], kernel_regularizer=regularizers.l2(1e-4),
                        kernel_initializer=self.initializer, activation=self.activation)(layer_1)
        layer_3 = Dense(units=self.input_shape[0]*self.num_filters[1], kernel_regularizer=regularizers.l2(1e-4),
                        kernel_initializer=self.initializer, activation=self.activation)(layer_2)
        output_shape = (self.batch_size, self.input_shape[0], self.num_filters[1])
        decoder_reshape = Reshape(output_shape[1:])(layer_3)
        decnn_layer_1 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[1],
                             kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(1e-4))(decoder_reshape)
        decnn_layer_2 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0],
                             kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(1e-4))(decnn_layer_1)
        deconv_final = Conv1D(kernel_initializer=self.initializer, activation='linear', padding="same",
                             filters=1, kernel_size=self.filter_length)(decnn_layer_2)

        y = CustomVariationalLayer()([input_tensor, deconv_final, mean_output, sigma_output])
        vae = Model(input_tensor, y)

        return vae

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_size), mean=0., stddev=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def compile(self):
        model = self.model()
        model.compile(loss=None, optimizer=self.optimizer)
        return model

    def train(self, x):
        if self.fallback_cpu is True:
            cpu_fallback()

        if self.limit_gpu_mem is not False:
            gpu_memory_manage()

        self.hyperparameter_writter()

        self.input_shape = (x.shape[1], 1,)
        self.outpot_shape = x.shape[1]

        csv_logger = CSVLogger(self.fullfilepath + 'log.csv', append=True, separator=',')

        if self.optimizer is None:
            self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.optimizer_epsilon,
                                  decay=0.0)

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=self.reduce_lr_epsilon,
                                      patience=self.reduce_lr_patience, min_lr=self.reduce_lr_min, mode='min', verbose=2)
        model = self.compile()

        try:
            plot_model(model, show_shapes=True, to_file=self.fullfilepath + 'model_{}.png'.format(self.runnum_name))
        except:
            print('Skipped plot_model! graphviz and pydot_ng are required to plot the model architecture')
            pass

        training_generator = DataGenerator(x.shape[1], self.batch_size).generate(x)

        model.fit_generator(generator=training_generator, steps_per_epoch=x.shape[0] // self.batch_size,
                            epochs=self.max_epochs, max_queue_size=20, verbose=2, workers=os.cpu_count())

        astronn_model = 'model.h5'

        model.save(self.fullfilepath + astronn_model)
        print(astronn_model + ' saved to {}'.format(self.fullfilepath + astronn_model))

        clear_session()

        return None


class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean, z_mean, z_log_var):
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = 7514 * metrics.binary_crossentropy(x, x_decoded_mean)
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