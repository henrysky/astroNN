# ---------------------------------------------------------#
#   astroNN.models.ApogeeBCNNCensored: Contain ApogeeBCNNCensored Model
# ---------------------------------------------------------#
from astroNN.apogee.plotting import ASPCAP_plots
from astroNN.config import keras_import_manager
from astroNN.models.BayesianCNNBase import BayesianCNNBase
from astroNN.nn.layers import MCDropout, StopGrad
from astroNN.nn.losses import mse_lin_wrapper, mse_var_wrapper
from astroNN.apogee import aspcap_mask
import tensorflow as tf

keras = keras_import_manager()
regularizers = keras.regularizers
MaxPooling1D, Conv1D, Dense, Flatten, Activation, Input = keras.layers.MaxPooling1D, keras.layers.Conv1D, \
                                                          keras.layers.Dense, keras.layers.Flatten, \
                                                          keras.layers.Activation, keras.layers.Input
concatenate = keras.layers.concatenate
Model = keras.models.Model


class ApogeeBCNNCensored(BayesianCNNBase, ASPCAP_plots):
    """
    Class for Bayesian censored convolutional neural network for stellar spectra analysis

    :History: 2018-May-27 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, lr=0.0005, dropout_rate=0.3):
        super().__init__()

        self._implementation_version = '1.0'
        self.initializer = 'RandomNormal'
        self.activation = 'relu'
        self.num_filters = [2, 4]
        self.filter_len = 8
        self.pool_length = 4
        self.num_hidden = [196, 96]
        self.max_epochs = 100
        self.lr = lr
        self.reduce_lr_epsilon = 0.00005

        self.reduce_lr_min = 1e-8
        self.reduce_lr_patience = 2
        self.l2 = 1e-7
        self.dropout_rate = dropout_rate

        self.input_norm_mode = 3

        self.task = 'regression'

        self.targetname = ['teff', 'logg', 'M', 'alpha', 'C', 'C1', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'K',
                           'Ca', 'Ti', 'Ti2', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'fakemag']

    def model(self):
        input_tensor = Input(shape=self._input_shape, name='input')
        labels_err_tensor = Input(shape=(self._labels_shape,), name='labels_err')

        # slice Al, Mg
        censored_c_input = tf.boolean_mask(input_tensor, aspcap_mask("C"))
        censored_al_input = tf.boolean_mask(input_tensor, aspcap_mask("Al"))
        censored_mg_input = tf.boolean_mask(input_tensor, aspcap_mask("Mg"))

        al_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout, )(Dense(unit=10)(censored_al_input))
        mg_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout, )(Dense(unit=10)(censored_mg_input))

        al_concat = Dense(unit=1)(concatenate([al_dense, StopGrad()(al_dense)]))
        mg_concat = Dense(unit=1)(concatenate([mg_dense, StopGrad()(mg_dense)]))

        al_concat_var = Dense(unit=1)(concatenate([al_dense, StopGrad()(al_dense)]))
        mg_concat_var = Dense(unit=1)(concatenate([mg_dense, StopGrad()(mg_dense)]))

        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, padding="same", filters=self.num_filters[0],
                             kernel_size=self.filter_len, kernel_regularizer=regularizers.l2(self.l2))(input_tensor)
        activation_1 = Activation(activation=self.activation)(cnn_layer_1)
        dropout_1 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(activation_1)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, padding="same", filters=self.num_filters[1],
                             kernel_size=self.filter_len, kernel_regularizer=regularizers.l2(self.l2))(dropout_1)
        activation_2 = Activation(activation=self.activation)(cnn_layer_2)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(activation_2)
        flattener = Flatten()(maxpool_1)
        dropout_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(flattener)
        layer_3 = Dense(units=self.num_hidden[0], kernel_regularizer=regularizers.l2(self.l2),
                        kernel_initializer=self.initializer,
                        activation=self.activation)(dropout_2)
        activation_3 = Activation(activation=self.activation)(layer_3)
        dropout_3 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(activation_3)
        layer_4 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l2(self.l2),
                        kernel_initializer=self.initializer,
                        activation=self.activation)(dropout_3)
        activation_4 = Activation(activation=self.activation)(layer_4)
        old_3_output = Dense(units=3)(activation_4)
        old_3_output_var = Dense(units=3)(activation_4)
        output = concatenate([old_3_output, al_concat, mg_concat], name='output')
        variance_output = concatenate([old_3_output_var, al_concat, mg_concat], name='variance_output')

        model = Model(inputs=[input_tensor, labels_err_tensor], outputs=[output, variance_output])
        # new astroNN high performance dropout variational inference on GPU expects single output
        model_prediction = Model(inputs=[input_tensor], outputs=concatenate([output, variance_output]))

        variance_loss = mse_var_wrapper(output, labels_err_tensor)
        output_loss = mse_lin_wrapper(variance_output, labels_err_tensor)

        return model, model_prediction, output_loss, variance_loss
