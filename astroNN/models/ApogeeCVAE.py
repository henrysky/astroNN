# ---------------------------------------------------------#
#   astroNN.models.ApogeeCVAE: Contain Variational Autoencoder Model
# ---------------------------------------------------------#
import tensorflow as tf

from astroNN.apogee.plotting import ASPCAP_plots
from astroNN.config import keras_import_manager
from astroNN.models.ConvVAEBase import ConvVAEBase
from astroNN.nn.layers import KLDivergenceLayer

keras = keras_import_manager()
regularizers = keras.regularizers
MaxPooling1D, Conv1D, Dense, Flatten, Activation, Input = keras.layers.MaxPooling1D, keras.layers.Conv1D, \
                                                          keras.layers.Dense, keras.layers.Flatten, \
                                                          keras.layers.Activation, keras.layers.Input
Dropout = keras.layers.Dropout
Lambda, Reshape, Multiply, Add = keras.layers.Lambda, keras.layers.Reshape, keras.layers.Multiply, keras.layers.Add
Model, Sequential = keras.models.Model, keras.models.Sequential


class ApogeeCVAE(ConvVAEBase, ASPCAP_plots):
    """
    NAME:
        ApogeeCVAE
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
        super().__init__()

        self._implementation_version = '1.0'
        self.batch_size = 64
        self.initializer = 'he_normal'
        self.activation = 'relu'
        self.optimizer = 'rmsprop'
        self.num_filters = [2, 4]
        self.filter_len = 8
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
        self.l1 = 1e-5
        self.l2 = 1e-5
        self.dropout_rate = 0.2

        self.input_norm_mode = 3
        self.labels_norm_mode = 3

    def model(self):
        input_tensor = Input(shape=self.input_shape, name='input')
        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0],
                             kernel_size=self.filter_len, kernel_regularizer=regularizers.l2(self.l2))(input_tensor)
        dropout_1 = Dropout(self.dropout_rate)(cnn_layer_1)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[1],
                             kernel_size=self.filter_len, kernel_regularizer=regularizers.l2(self.l2))(dropout_1)
        dropout_2 = Dropout(self.dropout_rate)(cnn_layer_2)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(dropout_2)
        flattener = Flatten()(maxpool_1)
        layer_4 = Dense(units=self.num_hidden[0], kernel_regularizer=regularizers.l1(self.l1),
                        kernel_initializer=self.initializer, activation=self.activation)(flattener)
        dropout_3 = Dropout(self.dropout_rate)(layer_4)
        layer_5 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l1(self.l1),
                        kernel_initializer=self.initializer, activation=self.activation)(dropout_3)
        dropout_4 = Dropout(self.dropout_rate)(layer_5)
        z_mu = Dense(units=self.latent_dim, activation="linear", name='mean_output',
                     kernel_regularizer=regularizers.l1(self.l1))(dropout_4)
        z_log_var = Dense(units=self.latent_dim, activation='linear', name='sigma_output',
                          kernel_regularizer=regularizers.l1(self.l1))(dropout_4)

        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        z_sigma = Lambda(lambda t: tf.exp(.5 * t))(z_log_var)

        eps = Input(tensor=tf.random_normal(stddev=1.0, shape=(tf.shape(input_tensor)[0], self.latent_dim)))
        z_eps = Multiply()([z_sigma, eps])
        z = Add()([z_mu, z_eps])

        decoder = Sequential()
        decoder.add(Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l1(self.l1),
                          kernel_initializer=self.initializer, activation=self.activation, input_dim=self.latent_dim))
        decoder.add(Dropout(self.dropout_rate))
        decoder.add(Dense(units=self.num_hidden[0], kernel_regularizer=regularizers.l1(self.l1),
                          kernel_initializer=self.initializer, activation=self.activation))
        decoder.add(Dropout(self.dropout_rate))
        decoder.add(Dense(units=self.input_shape[0] * self.num_filters[1], kernel_regularizer=regularizers.l2(self.l2),
                          kernel_initializer=self.initializer, activation=self.activation))
        decoder.add(Dropout(self.dropout_rate))
        output_shape = (self.batch_size, self.input_shape[0], self.num_filters[1])
        decoder.add(Reshape(output_shape[1:]))
        decoder.add(Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                           filters=self.num_filters[1],
                           kernel_size=self.filter_len, kernel_regularizer=regularizers.l2(self.l2)))
        decoder.add(Dropout(self.dropout_rate))
        decoder.add(Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                           filters=self.num_filters[0],
                           kernel_size=self.filter_len, kernel_regularizer=regularizers.l2(self.l2)))
        decoder.add(Dropout(self.dropout_rate))
        decoder.add(Conv1D(kernel_initializer=self.initializer, activation='linear', padding="same",
                           filters=1, kernel_size=self.filter_len, name='output'))

        x_pred = decoder(z)
        vae = Model(inputs=[input_tensor, eps], outputs=x_pred)
        encoder = Model(input_tensor, z_mu)

        return vae, encoder, decoder

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = tf.random_normal(shape=(tf.shape(z_mean)[0], self.latent_dim), mean=0., stddev=self.epsilon_std)
        return z_mean + tf.exp(z_log_var / 2) * epsilon
