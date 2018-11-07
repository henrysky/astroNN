# ---------------------------------------------------------#
#   astroNN.models.ApogeeBCNN: Contain ApogeeBCNN Model
# ---------------------------------------------------------#
from astroNN.config import keras_import_manager
from astroNN.models.BayesianCNNBase import BayesianCNNBase
from astroNN.nn.layers import MCDropout, StopGrad, BoolMask
from astroNN.nn.losses import mse_lin_wrapper, mse_var_wrapper
import tensorflow as tf
import numpy as np

keras = keras_import_manager()
regularizers = keras.regularizers
MaxPooling1D, Conv1D, Dense, Flatten, Activation, Input = keras.layers.MaxPooling1D, keras.layers.Conv1D, \
                                                          keras.layers.Dense, keras.layers.Flatten, \
                                                          keras.layers.Activation, keras.layers.Input
Lambda, Add, Subtract, Multiply = keras.layers.Lambda, keras.layers.Add, keras.layers.Subtract, keras.layers.Multiply
concatenate = keras.layers.concatenate
Model = keras.models.Model
RandomNormal = keras.initializers.RandomNormal


# noinspection PyCallingNonCallable
class ApogeeDR14GaiaDR2BCNN(BayesianCNNBase):
    """
    Class for Bayesian convolutional neural network for APOGEE DR14 Gaia DR2

    :History: 2018-Nov-06 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, lr=0.001, dropout_rate=0.3):
        super().__init__()

        self._implementation_version = '1.0'
        self.initializer = RandomNormal(mean=0.0, stddev=0.05)
        self.activation = 'relu'
        self.num_filters = [2, 4]
        self.filter_len = 8
        self.pool_length = 4
        self.num_hidden = [162, 64]
        self.max_epochs = 100
        self.lr = lr
        self.reduce_lr_epsilon = 0.00005

        self.reduce_lr_min = 1e-8
        self.reduce_lr_patience = 2
        self.l2 = 5e-9
        self.dropout_rate = dropout_rate

        self.input_norm_mode = 3

        self.task = 'regression'

        self.targetname = ['Ks-band fakemag']

    def inv_pow_mag(self,  mag):
        """
        take apparent magnitude tensor and return 10**(-0.2*mag)
        :return:
        """
        return StopGrad()(Lambda(lambda mag: tf.pow(tf.constant(10., dtype=tf.float32), tf.multiply(-0.2, mag)))(mag))

    def model(self):
        input_tensor = Input(shape=self._input_shape, name='input')  # training data
        labels_err_tensor = Input(shape=(self._labels_shape,), name='labels_err')

        specmask = np.zeros(self._input_shape[0], dtype=np.bool)
        specmask[:7515] = True  # mask to extract extinction correction apparent magnitude
        spectra = Lambda(lambda x: tf.expand_dims(x, axis=-1))(self.inv_pow_mag(BoolMask(specmask)(input_tensor)))

        magmask = np.zeros(self._input_shape[0], dtype=np.bool)
        magmask[7515] = True  # mask to extract extinction correction apparent magnitude
        # value to denorm magnitude
        mag_mean = tf.constant(self.input_mean[magmask], dtype=tf.float32)
        inv_pow_mag = Lambda(lambda x: tf.add(x, mag_mean))(self.inv_pow_mag(BoolMask(magmask)(input_tensor)))

        gaia_aux = np.zeros(self._input_shape[0], dtype=np.bool)
        gaia_aux[7516:] = True  # mask to extract data for gaia offset
        gaia_aux_data = BoolMask(magmask)(input_tensor)
        gaia_aux_hidden = MCDropout(self.dropout_rate, disable=self.disable_dropout)(Dense(units=18,
                                                                                           kernel_regularizer=regularizers.l2(self.l2),
                                                                                           kernel_initializer=self.initializer,
                                                                                           activation='tanh')(gaia_aux_data))
        offset = Dense(units=1, kernel_initializer=self.initializer, activation='tanh')(gaia_aux_hidden)

        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, padding="same", filters=self.num_filters[0],
                             kernel_size=self.filter_len, kernel_regularizer=regularizers.l2(self.l2))(spectra)
        activation_1 = Activation(activation=self.activation)(cnn_layer_1)
        dropout_1 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(activation_1)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, padding="same", filters=self.num_filters[1],
                             kernel_size=self.filter_len, kernel_regularizer=regularizers.l2(self.l2))(dropout_1)
        activation_2 = Activation(activation=self.activation)(cnn_layer_2)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(activation_2)
        flattener = Flatten()(maxpool_1)
        dropout_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(flattener)
        layer_3 = Dense(units=self.num_hidden[0], kernel_regularizer=regularizers.l2(self.l2),
                        kernel_initializer=self.initializer)(dropout_2)
        activation_3 = Activation(activation=self.activation)(layer_3)
        dropout_3 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(activation_3)
        layer_4 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l2(self.l2),
                        kernel_initializer=self.initializer)(dropout_3)
        activation_4 = Activation(activation=self.activation)(layer_4)
        fakemag_output = Dense(units=self._labels_shape, activation='softplus', name='fakemag_output')(activation_4)
        fakemag_variance_output = Dense(units=self._labels_shape, activation='linear',
                                        name='fakemag_variance_output')(activation_4)

        # multiplt a pre-determined de-normalization factor, such that fakemag std approx. 1 for Sloan APOGEE population
        _fakemag_denorm = Lambda(lambda x: tf.multiply(x, tf.constant(68, dtype=tf.float32)))(fakemag_output)
        _fakemag_var_denorm = Lambda(lambda x: tf.multiply(x, tf.constant(68, dtype=tf.float32)))(fakemag_variance_output)
        _fakemag_parallax = Multiply()([_fakemag_denorm, inv_pow_mag])

        output = Add(name='output')([_fakemag_parallax, offset])
        variance_output = Lambda(lambda x: tf.log(tf.abs(tf.multiply(x[2], tf.divide(tf.exp(x[0]), x[1])))),
                                 name='variance_output')([fakemag_variance_output, fakemag_output, _fakemag_parallax])

        model = Model(inputs=[input_tensor, labels_err_tensor], outputs=[output, variance_output])
        # new astroNN high performance dropout variational inference on GPU expects single output
        model_prediction = Model(inputs=[input_tensor], outputs=concatenate([_fakemag_denorm, _fakemag_var_denorm]))

        variance_loss = mse_var_wrapper(output, labels_err_tensor)
        output_loss = mse_lin_wrapper(variance_output, labels_err_tensor)

        return model, model_prediction, output_loss, variance_loss
