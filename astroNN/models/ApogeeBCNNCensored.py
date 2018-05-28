# ---------------------------------------------------------#
#   astroNN.models.ApogeeBCNNCensored: Contain ApogeeBCNNCensored Model
# ---------------------------------------------------------#
from astroNN.apogee.plotting import ASPCAP_plots
from astroNN.config import keras_import_manager
from astroNN.models.BayesianCNNBase import BayesianCNNBase
from astroNN.nn.layers import MCDropout, StopGrad, BoolMask
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
    [Testing purpose only] Class for Bayesian censored convolutional neural network for stellar spectra analysis

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

        self.targetname = ['teff', 'logg', 'Fe', 'C', 'C1', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'K',
                           'Ca', 'Ti', 'Ti2', 'V', 'Cr', 'Mn', 'Co', 'Ni']

    def model(self):
        input_tensor = Input(shape=self._input_shape, name='input')
        labels_err_tensor = Input(shape=(self._labels_shape,), name='labels_err')

        # slice spectra to censor out useless region for elements
        censored_c_input = BoolMask(aspcap_mask("C", dr=14))(Flatten()(input_tensor))
        censored_c1_input = BoolMask(aspcap_mask("C1", dr=14))(Flatten()(input_tensor))
        censored_n_input = BoolMask(aspcap_mask("N", dr=14))(Flatten()(input_tensor))
        censored_o_input = BoolMask(aspcap_mask("O", dr=14))(Flatten()(input_tensor))
        censored_na_input = BoolMask(aspcap_mask("Na", dr=14))(Flatten()(input_tensor))
        censored_mg_input = BoolMask(aspcap_mask("Mg", dr=14))(Flatten()(input_tensor))
        censored_al_input = BoolMask(aspcap_mask("Al", dr=14))(Flatten()(input_tensor))
        censored_si_input = BoolMask(aspcap_mask("Si", dr=14))(Flatten()(input_tensor))
        censored_p_input = BoolMask(aspcap_mask("P", dr=14))(Flatten()(input_tensor))
        censored_s_input = BoolMask(aspcap_mask("S", dr=14))(Flatten()(input_tensor))
        censored_k_input = BoolMask(aspcap_mask("K", dr=14))(Flatten()(input_tensor))
        censored_ca_input = BoolMask(aspcap_mask("Ca", dr=14))(Flatten()(input_tensor))
        censored_ti_input = BoolMask(aspcap_mask("Ti", dr=14))(Flatten()(input_tensor))
        censored_ti2_input = BoolMask(aspcap_mask("Ti2", dr=14))(Flatten()(input_tensor))
        censored_v_input = BoolMask(aspcap_mask("V", dr=14))(Flatten()(input_tensor))
        censored_cr_input = BoolMask(aspcap_mask("Cr", dr=14))(Flatten()(input_tensor))
        censored_mn_input = BoolMask(aspcap_mask("Mn", dr=14))(Flatten()(input_tensor))
        censored_co_input = BoolMask(aspcap_mask("Co", dr=14))(Flatten()(input_tensor))
        censored_ni_input = BoolMask(aspcap_mask("Ni", dr=14))(Flatten()(input_tensor))

        # get 11 neurones from each elements from censored spectra
        c_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(Dense(units=10)(censored_c_input))
        c1_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(Dense(units=10)(censored_c1_input))
        n_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(Dense(units=10)(censored_n_input))
        o_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(Dense(units=10)(censored_o_input))
        na_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(Dense(units=10)(censored_na_input))
        mg_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(Dense(units=10)(censored_mg_input))
        al_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(Dense(units=10)(censored_al_input))
        si_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(Dense(units=10)(censored_si_input))
        p_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(Dense(units=10)(censored_p_input))
        s_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(Dense(units=10)(censored_s_input))
        k_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(Dense(units=10)(censored_k_input))
        ca_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(Dense(units=10)(censored_ca_input))
        ti_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(Dense(units=10)(censored_ti_input))
        ti2_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(Dense(units=10)(censored_ti2_input))
        v_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(Dense(units=10)(censored_v_input))
        cr_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(Dense(units=10)(censored_cr_input))
        mn_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(Dense(units=10)(censored_mn_input))
        co_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(Dense(units=10)(censored_co_input))
        ni_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(Dense(units=10)(censored_ni_input))

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
        old_3_output = Dense(units=3, kernel_regularizer=regularizers.l2(self.l2),
                             kernel_initializer=self.initializer)(activation_4)
        old_3_output_var = Dense(units=3)(activation_4)

        # get the final answer
        c_concat = Dense(units=1)(concatenate([c_dense, StopGrad()(old_3_output)]))
        c1_concat = Dense(units=1)(concatenate([c1_dense, StopGrad()(old_3_output)]))
        n_concat = Dense(units=1)(concatenate([n_dense, StopGrad()(old_3_output)]))
        o_concat = Dense(units=1)(concatenate([o_dense, StopGrad()(old_3_output)]))
        na_concat = Dense(units=1)(concatenate([na_dense, StopGrad()(old_3_output)]))
        mg_concat = Dense(units=1)(concatenate([mg_dense, StopGrad()(old_3_output)]))
        al_concat = Dense(units=1)(concatenate([al_dense, StopGrad()(old_3_output)]))
        si_concat = Dense(units=1)(concatenate([si_dense, StopGrad()(old_3_output)]))
        p_concat = Dense(units=1)(concatenate([p_dense, StopGrad()(old_3_output)]))
        s_concat = Dense(units=1)(concatenate([s_dense, StopGrad()(old_3_output)]))
        k_concat = Dense(units=1)(concatenate([k_dense, StopGrad()(old_3_output)]))
        ca_concat = Dense(units=1)(concatenate([ca_dense, StopGrad()(old_3_output)]))
        ti_concat = Dense(units=1)(concatenate([ti_dense, StopGrad()(old_3_output)]))
        ti2_concat = Dense(units=1)(concatenate([ti2_dense, StopGrad()(old_3_output)]))
        v_concat = Dense(units=1)(concatenate([v_dense, StopGrad()(old_3_output)]))
        cr_concat = Dense(units=1)(concatenate([cr_dense, StopGrad()(old_3_output)]))
        mn_concat = Dense(units=1)(concatenate([mn_dense, StopGrad()(old_3_output)]))
        co_concat = Dense(units=1)(concatenate([co_dense, StopGrad()(old_3_output)]))
        ni_concat = Dense(units=1)(concatenate([ni_dense, StopGrad()(old_3_output)]))

        # get the final predictive uncertainty
        c_concat_var = Dense(units=1)(concatenate([c_dense, StopGrad()(old_3_output)]))
        c1_concat_var = Dense(units=1)(concatenate([c1_dense, StopGrad()(old_3_output)]))
        n_concat_var = Dense(units=1)(concatenate([n_dense, StopGrad()(old_3_output)]))
        o_concat_var = Dense(units=1)(concatenate([o_dense, StopGrad()(old_3_output)]))
        na_concat_var = Dense(units=1)(concatenate([na_dense, StopGrad()(old_3_output)]))
        mg_concat_var = Dense(units=1)(concatenate([mg_dense, StopGrad()(old_3_output)]))
        al_concat_var = Dense(units=1)(concatenate([al_dense, StopGrad()(old_3_output)]))
        si_concat_var = Dense(units=1)(concatenate([si_dense, StopGrad()(old_3_output)]))
        p_concat_var = Dense(units=1)(concatenate([p_dense, StopGrad()(old_3_output)]))
        s_concat_var = Dense(units=1)(concatenate([s_dense, StopGrad()(old_3_output)]))
        k_concat_var = Dense(units=1)(concatenate([k_dense, StopGrad()(old_3_output)]))
        ca_concat_var = Dense(units=1)(concatenate([ca_dense, StopGrad()(old_3_output)]))
        ti_concat_var = Dense(units=1)(concatenate([ti_dense, StopGrad()(old_3_output)]))
        ti2_concat_var = Dense(units=1)(concatenate([ti2_dense, StopGrad()(old_3_output)]))
        v_concat_var = Dense(units=1)(concatenate([v_dense, StopGrad()(old_3_output)]))
        cr_concat_var = Dense(units=1)(concatenate([cr_dense, StopGrad()(old_3_output)]))
        mn_concat_var = Dense(units=1)(concatenate([mn_dense, StopGrad()(old_3_output)]))
        co_concat_var = Dense(units=1)(concatenate([co_dense, StopGrad()(old_3_output)]))
        ni_concat_var = Dense(units=1)(concatenate([ni_dense, StopGrad()(old_3_output)]))

        # concatenate answer
        output = concatenate([old_3_output, c_concat, c1_concat, n_concat, o_concat, na_concat, mg_concat, al_concat,
                              si_concat, p_concat, s_concat, k_concat, ca_concat, ti_concat, ti2_concat, v_concat,
                              cr_concat, mn_concat, co_concat, ni_concat], name='output')

        # concatenate predictive uncertainty
        variance_output = concatenate([old_3_output_var, c_concat_var, c1_concat_var, n_concat_var, o_concat_var,
                                       na_concat_var, mg_concat_var, al_concat_var, si_concat_var, p_concat_var,
                                       s_concat_var, k_concat_var, ca_concat_var, ti_concat_var, ti2_concat_var,
                                       v_concat_var, cr_concat_var, mn_concat_var, co_concat_var, ni_concat_var],
                                      name='variance_output')

        model = Model(inputs=[input_tensor, labels_err_tensor], outputs=[output, variance_output])
        # new astroNN high performance dropout variational inference on GPU expects single output
        model_prediction = Model(inputs=input_tensor, outputs=concatenate([output, variance_output]))

        variance_loss = mse_var_wrapper(output, labels_err_tensor)
        output_loss = mse_lin_wrapper(variance_output, labels_err_tensor)

        return model, model_prediction, output_loss, variance_loss
