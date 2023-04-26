# ---------------------------------------------------------#
#   astroNN.models.apogee_models: Contain Apogee Models
# ---------------------------------------------------------#
import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk

from astroNN.apogee import aspcap_mask
from astroNN.models.base_bayesian_cnn import BayesianCNNBase
from astroNN.models.base_cnn import CNNBase
from astroNN.models.base_vae import ConvVAEBase
from astroNN.nn.layers import (
    MCDropout,
    BoolMask,
    StopGrad,
    KLDivergenceLayer,
    TensorInput,
    VAESampling,
)
from astroNN.nn.losses import (
    bayesian_binary_crossentropy_wrapper,
    bayesian_binary_crossentropy_var_wrapper,
)
from astroNN.nn.losses import (
    bayesian_categorical_crossentropy_wrapper,
    bayesian_categorical_crossentropy_var_wrapper,
)
from astroNN.nn.losses import mse_lin_wrapper, mse_var_wrapper

Add = tfk.layers.Add
Dense = tfk.layers.Dense
Input = tfk.layers.Input
Conv1D = tfk.layers.Conv1D
Conv2D = tfk.layers.Conv2D
Lambda = tfk.layers.Lambda
Reshape = tfk.layers.Reshape
Dropout = tfk.layers.Dropout
Flatten = tfk.layers.Flatten
Multiply = tfk.layers.Multiply
Activation = tfk.layers.Activation
concatenate = tfk.layers.concatenate
MaxPooling1D = tfk.layers.MaxPooling1D
MaxPooling2D = tfk.layers.MaxPooling2D
Conv1DTranspose = tfk.layers.Conv1DTranspose

Model = tfk.models.Model
Sequential = tfk.models.Sequential

regularizers = tfk.regularizers
MaxNorm = tfk.constraints.MaxNorm
RandomNormal = tfk.initializers.RandomNormal


# noinspection PyCallingNonCallable
class ApogeeBCNN(BayesianCNNBase):
    """
    Class for Bayesian convolutional neural network for stellar spectra analysis

    :History: 2017-Dec-21 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, lr=0.0005, dropout_rate=0.3):
        super().__init__()

        self._implementation_version = "1.0"
        self.initializer = RandomNormal(mean=0.0, stddev=0.05)
        self.activation = "relu"
        self.num_filters = [2, 4]
        self.filter_len = 8
        self.pool_length = 4
        self.num_hidden = [196, 96]
        self.max_epochs = 100
        self.lr = lr
        self.reduce_lr_epsilon = 0.00005

        self.reduce_lr_min = 1e-8
        self.reduce_lr_patience = 2
        self.l2 = 5e-9
        self.dropout_rate = dropout_rate

        self.input_norm_mode = 3

        self.task = "regression"

        self.targetname = [
            "teff",
            "logg",
            "M",
            "alpha",
            "C",
            "C1",
            "N",
            "O",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "K",
            "Ca",
            "Ti",
            "Ti2",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "fakemag",
        ]

    def model(self):
        input_tensor = Input(shape=self._input_shape["input"], name="input")
        labels_err_tensor = Input(
            shape=(self._labels_shape["output"],), name="labels_err"
        )

        cnn_layer_1 = Conv1D(
            kernel_initializer=self.initializer,
            padding="same",
            filters=self.num_filters[0],
            kernel_size=self.filter_len,
            kernel_regularizer=regularizers.l2(self.l2),
        )(input_tensor)
        activation_1 = Activation(activation=self.activation)(cnn_layer_1)
        dropout_1 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            activation_1
        )
        cnn_layer_2 = Conv1D(
            kernel_initializer=self.initializer,
            padding="same",
            filters=self.num_filters[1],
            kernel_size=self.filter_len,
            kernel_regularizer=regularizers.l2(self.l2),
        )(dropout_1)
        activation_2 = Activation(activation=self.activation)(cnn_layer_2)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(activation_2)
        flattener = Flatten()(maxpool_1)
        dropout_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            flattener
        )
        layer_3 = Dense(
            units=self.num_hidden[0],
            kernel_regularizer=regularizers.l2(self.l2),
            kernel_initializer=self.initializer,
        )(dropout_2)
        activation_3 = Activation(activation=self.activation)(layer_3)
        dropout_3 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            activation_3
        )
        layer_4 = Dense(
            units=self.num_hidden[1],
            kernel_regularizer=regularizers.l2(self.l2),
            kernel_initializer=self.initializer,
        )(dropout_3)
        activation_4 = Activation(activation=self.activation)(layer_4)
        output = Dense(
            units=self._labels_shape["output"],
            activation=self._last_layer_activation,
            name="output",
        )(activation_4)
        variance_output = Dense(
            units=self._labels_shape["output"],
            activation="linear",
            name="variance_output",
        )(activation_4)

        model = Model(
            inputs=[input_tensor, labels_err_tensor], outputs=[output, variance_output]
        )
        # new astroNN high performance dropout variational inference on GPU expects single output
        model_prediction = Model(
            inputs=[input_tensor], outputs=concatenate([output, variance_output])
        )

        if self.task == "regression":
            variance_loss = mse_var_wrapper(output, labels_err_tensor)
            output_loss = mse_lin_wrapper(variance_output, labels_err_tensor)
        elif self.task == "classification":
            output_loss = bayesian_categorical_crossentropy_wrapper(variance_output)
            variance_loss = bayesian_categorical_crossentropy_var_wrapper(output)
        elif self.task == "binary_classification":
            output_loss = bayesian_binary_crossentropy_wrapper(variance_output)
            variance_loss = bayesian_binary_crossentropy_var_wrapper(output)
        else:
            raise RuntimeError(
                "Only 'regression', 'classification' and 'binary_classification' are supported"
            )

        return model, model_prediction, output_loss, variance_loss


# noinspection PyCallingNonCallable
class ApogeeBCNNCensored(BayesianCNNBase):
    """
    Class for Bayesian censored convolutional neural network for stellar spectra analysis [specifically APOGEE
    DR14 spectra only]

    Described in the paper: https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.3255L/abstract

    :History: 2018-May-27 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, lr=0.0005, dropout_rate=0.3):
        super().__init__()

        self._implementation_version = "1.0"
        self.initializer = RandomNormal(mean=0.0, stddev=0.05)
        self.activation = "relu"
        self.num_filters = [2, 4]
        self.filter_len = 8
        self.pool_length = 4
        # number of neurone for [ApogeeBCNN_Dense_1, ApogeeBCNN_Dense_2, aspcap_1, aspcap_2, hidden]
        self.num_hidden = [192, 96, 32, 16, 2]
        self.max_epochs = 50
        self.lr = lr
        self.reduce_lr_epsilon = 0.00005
        self.maxnorm = 0.5

        self.reduce_lr_min = 1e-8
        self.reduce_lr_patience = 2
        self.l2 = 5e-9
        self.dropout_rate = dropout_rate

        self.input_norm_mode = 3

        self.task = "regression"

        self.targetname = [
            "teff",
            "logg",
            "C",
            "C1",
            "N",
            "O",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "K",
            "Ca",
            "Ti",
            "Ti2",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
        ]

    def model(self):
        input_tensor = Input(shape=self._input_shape["input"], name="input")
        input_tensor_flattened = Flatten()(input_tensor)
        labels_err_tensor = Input(
            shape=(self._labels_shape["output"],), name="labels_err"
        )

        # slice spectra to censor out useless region for elements
        censored_c_input = BoolMask(aspcap_mask("C", dr=14), name="C_Mask")(
            input_tensor_flattened
        )
        censored_c1_input = BoolMask(aspcap_mask("C1", dr=14), name="C1_Mask")(
            input_tensor_flattened
        )
        censored_n_input = BoolMask(aspcap_mask("N", dr=14), name="N_Mask")(
            input_tensor_flattened
        )
        censored_o_input = BoolMask(aspcap_mask("O", dr=14), name="O_Mask")(
            input_tensor_flattened
        )
        censored_na_input = BoolMask(aspcap_mask("Na", dr=14), name="Na_Mask")(
            input_tensor_flattened
        )
        censored_mg_input = BoolMask(aspcap_mask("Mg", dr=14), name="Mg_Mask")(
            input_tensor_flattened
        )
        censored_al_input = BoolMask(aspcap_mask("Al", dr=14), name="Al_Mask")(
            input_tensor_flattened
        )
        censored_si_input = BoolMask(aspcap_mask("Si", dr=14), name="Si_Mask")(
            input_tensor_flattened
        )
        censored_p_input = BoolMask(aspcap_mask("P", dr=14), name="P_Mask")(
            input_tensor_flattened
        )
        censored_s_input = BoolMask(aspcap_mask("S", dr=14), name="S_Mask")(
            input_tensor_flattened
        )
        censored_k_input = BoolMask(aspcap_mask("K", dr=14), name="K_Mask")(
            input_tensor_flattened
        )
        censored_ca_input = BoolMask(aspcap_mask("Ca", dr=14), name="Ca_Mask")(
            input_tensor_flattened
        )
        censored_ti_input = BoolMask(aspcap_mask("Ti", dr=14), name="Ti_Mask")(
            input_tensor_flattened
        )
        censored_ti2_input = BoolMask(aspcap_mask("Ti2", dr=14), name="Ti2_Mask")(
            input_tensor_flattened
        )
        censored_v_input = BoolMask(aspcap_mask("V", dr=14), name="V_Mask")(
            input_tensor_flattened
        )
        censored_cr_input = BoolMask(aspcap_mask("Cr", dr=14), name="Cr_Mask")(
            input_tensor_flattened
        )
        censored_mn_input = BoolMask(aspcap_mask("Mn", dr=14), name="Mn_Mask")(
            input_tensor_flattened
        )
        censored_co_input = BoolMask(aspcap_mask("Co", dr=14), name="Co_Mask")(
            input_tensor_flattened
        )
        censored_ni_input = BoolMask(aspcap_mask("Ni", dr=14), name="Ni_Mask")(
            input_tensor_flattened
        )

        # get neurones from each elements from censored spectra
        c_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[2] * 8,
                kernel_initializer=self.initializer,
                name="c_dense",
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2),
            )(censored_c_input)
        )
        c1_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[2],
                kernel_initializer=self.initializer,
                name="c1_dense",
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2),
            )(censored_c1_input)
        )
        n_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[2] * 8,
                kernel_initializer=self.initializer,
                name="n_dense",
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2),
            )(censored_n_input)
        )
        o_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[2],
                kernel_initializer=self.initializer,
                name="o_dense",
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2),
            )(censored_o_input)
        )
        na_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[2],
                kernel_initializer=self.initializer,
                name="na_dense",
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2),
            )(censored_na_input)
        )
        mg_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[2],
                kernel_initializer=self.initializer,
                name="mg_dense",
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2),
            )(censored_mg_input)
        )
        al_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[2],
                kernel_initializer=self.initializer,
                name="al_dense",
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2),
            )(censored_al_input)
        )
        si_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[2],
                kernel_initializer=self.initializer,
                name="si_dense",
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2),
            )(censored_si_input)
        )
        p_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[2],
                kernel_initializer=self.initializer,
                name="p_dense",
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2),
            )(censored_p_input)
        )
        s_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[2],
                kernel_initializer=self.initializer,
                name="s_dense",
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2),
            )(censored_s_input)
        )
        k_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[2],
                kernel_initializer=self.initializer,
                name="k_dense",
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2),
            )(censored_k_input)
        )
        ca_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[2],
                kernel_initializer=self.initializer,
                name="ca_dense",
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2),
            )(censored_ca_input)
        )
        ti_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[2],
                kernel_initializer=self.initializer,
                name="ti_dense",
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2),
            )(censored_ti_input)
        )
        ti2_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[2],
                kernel_initializer=self.initializer,
                name="ti2_dense",
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2),
            )(censored_ti2_input)
        )
        v_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[2],
                kernel_initializer=self.initializer,
                name="v_dense",
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2),
            )(censored_v_input)
        )
        cr_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[2],
                kernel_initializer=self.initializer,
                name="cr_dense",
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2),
            )(censored_cr_input)
        )
        mn_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[2],
                kernel_initializer=self.initializer,
                name="mn_dense",
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2),
            )(censored_mn_input)
        )
        co_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[2],
                kernel_initializer=self.initializer,
                name="co_dense",
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2),
            )(censored_co_input)
        )
        ni_dense = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[2],
                kernel_initializer=self.initializer,
                name="ni_dense",
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2),
            )(censored_ni_input)
        )

        # get neurones from each elements from censored spectra
        c_dense_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[3] * 4,
                kernel_initializer=self.initializer,
                activation=self.activation,
                name="c_dense_2",
            )(c_dense)
        )
        c1_dense_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[3],
                kernel_initializer=self.initializer,
                activation=self.activation,
                name="c1_dense_2",
            )(c1_dense)
        )
        n_dense_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[3] * 4,
                kernel_initializer=self.initializer,
                activation=self.activation,
                name="n_dense_2",
            )(n_dense)
        )
        o_dense_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[3],
                kernel_initializer=self.initializer,
                activation=self.activation,
                name="o_dense_2",
            )(o_dense)
        )
        na_dense_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[3],
                kernel_initializer=self.initializer,
                activation=self.activation,
                name="na_dense_2",
            )(na_dense)
        )
        mg_dense_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[3],
                kernel_initializer=self.initializer,
                activation=self.activation,
                name="mg_dense_2",
            )(mg_dense)
        )
        al_dense_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[3],
                kernel_initializer=self.initializer,
                activation=self.activation,
                name="al_dense_2",
            )(al_dense)
        )
        si_dense_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[3],
                kernel_initializer=self.initializer,
                activation=self.activation,
                name="si_dense_2",
            )(si_dense)
        )
        p_dense_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[3],
                kernel_initializer=self.initializer,
                activation=self.activation,
                name="p_dense_2",
            )(p_dense)
        )
        s_dense_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[3],
                kernel_initializer=self.initializer,
                activation=self.activation,
                name="s_dense_2",
            )(s_dense)
        )
        k_dense_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[3],
                kernel_initializer=self.initializer,
                activation=self.activation,
                name="k_dense_2",
            )(k_dense)
        )
        ca_dense_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[3],
                kernel_initializer=self.initializer,
                activation=self.activation,
                name="ca_dense_2",
            )(ca_dense)
        )
        ti_dense_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[3],
                kernel_initializer=self.initializer,
                activation=self.activation,
                name="ti_dense_2",
            )(ti_dense)
        )
        ti2_dense_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[3],
                kernel_initializer=self.initializer,
                activation=self.activation,
                name="ti2_dense_2",
            )(ti2_dense)
        )
        v_dense_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[3],
                kernel_initializer=self.initializer,
                activation=self.activation,
                name="v_dense_2",
            )(v_dense)
        )
        cr_dense_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[3],
                kernel_initializer=self.initializer,
                activation=self.activation,
                name="cr_dense_2",
            )(cr_dense)
        )
        mn_dense_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[3],
                kernel_initializer=self.initializer,
                activation=self.activation,
                name="mn_dense_2",
            )(mn_dense)
        )
        co_dense_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[3],
                kernel_initializer=self.initializer,
                activation=self.activation,
                name="co_dense_2",
            )(co_dense)
        )
        ni_dense_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[3],
                kernel_initializer=self.initializer,
                activation=self.activation,
                name="ni_dense_2",
            )(ni_dense)
        )

        # Basically the same as ApogeeBCNN structure
        cnn_layer_1 = Conv1D(
            kernel_initializer=self.initializer,
            padding="same",
            filters=self.num_filters[0],
            kernel_size=self.filter_len,
            kernel_regularizer=regularizers.l2(self.l2),
        )(input_tensor)
        activation_1 = Activation(activation=self.activation)(cnn_layer_1)
        dropout_1 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            activation_1
        )
        cnn_layer_2 = Conv1D(
            kernel_initializer=self.initializer,
            padding="same",
            filters=self.num_filters[1],
            kernel_size=self.filter_len,
            kernel_regularizer=regularizers.l2(self.l2),
        )(dropout_1)
        activation_2 = Activation(activation=self.activation)(cnn_layer_2)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(activation_2)
        flattener = Flatten()(maxpool_1)
        dropout_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            flattener
        )
        layer_3 = Dense(
            units=self.num_hidden[0],
            kernel_regularizer=regularizers.l2(self.l2),
            kernel_initializer=self.initializer,
        )(dropout_2)
        activation_3 = Activation(activation=self.activation)(layer_3)
        dropout_3 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            activation_3
        )
        layer_4 = Dense(
            units=self.num_hidden[1],
            kernel_regularizer=regularizers.l2(self.l2),
            kernel_initializer=self.initializer,
        )(dropout_3)
        activation_4 = Activation(activation=self.activation)(layer_4)
        teff_output = Dense(units=1)(activation_4)
        logg_output = Dense(units=1)(activation_4)
        fe_output = Dense(units=1)(activation_4)
        old_3_output_wo_grad = StopGrad()(
            concatenate([teff_output, logg_output, fe_output])
        )

        teff_output_var = Dense(units=1)(activation_4)
        logg_output_var = Dense(units=1)(activation_4)
        fe_output_var = Dense(units=1)(activation_4)

        aux_fullspec = Dense(
            units=self.num_hidden[4],
            kernel_initializer=self.initializer,
            kernel_constraint=MaxNorm(self.maxnorm),
            name="aux_fullspec",
        )(activation_4)

        fullspec_hidden = concatenate([aux_fullspec, old_3_output_wo_grad])

        # get the final answer
        c_concat = Dense(units=1, name="c_concat")(
            concatenate([c_dense_2, fullspec_hidden])
        )
        c1_concat = Dense(units=1, name="c1_concat")(
            concatenate([c1_dense_2, fullspec_hidden])
        )
        n_concat = Dense(units=1, name="n_concat")(
            concatenate([n_dense_2, fullspec_hidden])
        )
        o_concat = Dense(units=1, name="o_concat")(
            concatenate([o_dense_2, fullspec_hidden])
        )
        na_concat = Dense(units=1, name="na_concat")(
            concatenate([na_dense_2, fullspec_hidden])
        )
        mg_concat = Dense(units=1, name="mg_concat")(
            concatenate([mg_dense_2, fullspec_hidden])
        )
        al_concat = Dense(units=1, name="al_concat")(
            concatenate([al_dense_2, fullspec_hidden])
        )
        si_concat = Dense(units=1, name="si_concat")(
            concatenate([si_dense_2, fullspec_hidden])
        )
        p_concat = Dense(units=1, name="p_concat")(
            concatenate([p_dense_2, fullspec_hidden])
        )
        s_concat = Dense(units=1, name="s_concat")(
            concatenate([s_dense_2, fullspec_hidden])
        )
        k_concat = Dense(units=1, name="k_concat")(
            concatenate([k_dense_2, fullspec_hidden])
        )
        ca_concat = Dense(units=1, name="ca_concat")(
            concatenate([ca_dense_2, fullspec_hidden])
        )
        ti_concat = Dense(units=1, name="ti_concat")(
            concatenate([ti_dense_2, fullspec_hidden])
        )
        ti2_concat = Dense(units=1, name="ti2_concat")(
            concatenate([ti2_dense_2, fullspec_hidden])
        )
        v_concat = Dense(units=1, name="v_concat")(
            concatenate([v_dense_2, fullspec_hidden])
        )
        cr_concat = Dense(units=1, name="cr_concat")(
            concatenate([cr_dense_2, fullspec_hidden])
        )
        mn_concat = Dense(units=1, name="mn_concat")(
            concatenate([mn_dense_2, fullspec_hidden])
        )
        co_concat = Dense(units=1, name="co_concat")(
            concatenate([co_dense_2, fullspec_hidden])
        )
        ni_concat = Dense(units=1, name="ni_concat")(
            concatenate([ni_dense_2, fullspec_hidden])
        )

        # get the final predictive uncertainty
        c_concat_var = Dense(units=1, name="c_concat_var")(
            concatenate([c_dense_2, fullspec_hidden])
        )
        c1_concat_var = Dense(units=1, name="c1_concat_var")(
            concatenate([c1_dense_2, fullspec_hidden])
        )
        n_concat_var = Dense(units=1, name="n_concat_var")(
            concatenate([n_dense_2, fullspec_hidden])
        )
        o_concat_var = Dense(units=1, name="o_concat_var")(
            concatenate([o_dense_2, fullspec_hidden])
        )
        na_concat_var = Dense(units=1, name="na_concat_var")(
            concatenate([na_dense_2, fullspec_hidden])
        )
        mg_concat_var = Dense(units=1, name="mg_concat_var")(
            concatenate([mg_dense_2, fullspec_hidden])
        )
        al_concat_var = Dense(units=1, name="al_concat_var")(
            concatenate([al_dense_2, fullspec_hidden])
        )
        si_concat_var = Dense(units=1, name="si_concat_var")(
            concatenate([si_dense_2, fullspec_hidden])
        )
        p_concat_var = Dense(units=1, name="p_concat_var")(
            concatenate([p_dense_2, fullspec_hidden])
        )
        s_concat_var = Dense(units=1, name="s_concat_var")(
            concatenate([s_dense_2, fullspec_hidden])
        )
        k_concat_var = Dense(units=1, name="k_concat_var")(
            concatenate([k_dense_2, fullspec_hidden])
        )
        ca_concat_var = Dense(units=1, name="ca_concat_var")(
            concatenate([ca_dense_2, fullspec_hidden])
        )
        ti_concat_var = Dense(units=1, name="ti_concat_var")(
            concatenate([ti_dense_2, fullspec_hidden])
        )
        ti2_concat_var = Dense(units=1, name="ti2_concat_var")(
            concatenate([ti2_dense_2, fullspec_hidden])
        )
        v_concat_var = Dense(units=1, name="v_concat_var")(
            concatenate([v_dense_2, fullspec_hidden])
        )
        cr_concat_var = Dense(units=1, name="cr_concat_var")(
            concatenate([cr_dense_2, fullspec_hidden])
        )
        mn_concat_var = Dense(units=1, name="mn_concat_var")(
            concatenate([mn_dense_2, fullspec_hidden])
        )
        co_concat_var = Dense(units=1, name="co_concat_var")(
            concatenate([co_dense_2, fullspec_hidden])
        )
        ni_concat_var = Dense(units=1, name="ni_concat_var")(
            concatenate([ni_dense_2, fullspec_hidden])
        )

        # concatenate answer
        output = concatenate(
            [
                teff_output,
                logg_output,
                c_concat,
                c1_concat,
                n_concat,
                o_concat,
                na_concat,
                mg_concat,
                al_concat,
                si_concat,
                p_concat,
                s_concat,
                k_concat,
                ca_concat,
                ti_concat,
                ti2_concat,
                v_concat,
                cr_concat,
                mn_concat,
                fe_output,
                co_concat,
                ni_concat,
            ],
            name="output",
        )

        # concatenate predictive uncertainty
        variance_output = concatenate(
            [
                teff_output_var,
                logg_output_var,
                c_concat_var,
                c1_concat_var,
                n_concat_var,
                o_concat_var,
                na_concat_var,
                mg_concat_var,
                al_concat_var,
                si_concat_var,
                p_concat_var,
                s_concat_var,
                k_concat_var,
                ca_concat_var,
                ti_concat_var,
                ti2_concat_var,
                v_concat_var,
                cr_concat_var,
                mn_concat_var,
                fe_output_var,
                co_concat_var,
                ni_concat_var,
            ],
            name="variance_output",
        )

        model = Model(
            inputs=[input_tensor, labels_err_tensor], outputs=[output, variance_output]
        )
        # new astroNN high performance dropout variational inference on GPU expects single output
        model_prediction = Model(
            inputs=input_tensor, outputs=concatenate([output, variance_output])
        )

        variance_loss = mse_var_wrapper(output, labels_err_tensor)
        output_loss = mse_lin_wrapper(variance_output, labels_err_tensor)

        return model, model_prediction, output_loss, variance_loss


class ApogeeCNN(CNNBase):
    """
    Class for Convolutional Neural Network for stellar spectra analysis

    :History: 2017-Dec-21 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, lr=0.005):
        super().__init__()

        self._implementation_version = "1.0"
        self.initializer = "he_normal"
        self.activation = "relu"
        self.num_filters = [2, 4]
        self.filter_len = 8
        self.pool_length = 4
        self.num_hidden = [196, 96]
        self.max_epochs = 100
        self.lr = lr
        self.reduce_lr_epsilon = 0.00005

        self.reduce_lr_min = 1e-8
        self.reduce_lr_patience = 2
        self.l2 = 1e-5
        self.dropout_rate = 0.1

        self.input_norm_mode = 3

        self.task = "regression"
        self.targetname = [
            "teff",
            "logg",
            "M",
            "alpha",
            "C",
            "C1",
            "N",
            "O",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "K",
            "Ca",
            "Ti",
            "Ti2",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "fakemag",
        ]

    def model(self):
        input_tensor = Input(shape=self._input_shape["input"], name="input")
        cnn_layer_1 = Conv1D(
            kernel_initializer=self.initializer,
            padding="same",
            filters=self.num_filters[0],
            kernel_size=self.filter_len,
            kernel_regularizer=regularizers.l2(self.l2),
        )(input_tensor)
        activation_1 = Activation(activation=self.activation)(cnn_layer_1)
        cnn_layer_2 = Conv1D(
            kernel_initializer=self.initializer,
            padding="same",
            filters=self.num_filters[1],
            kernel_size=self.filter_len,
            kernel_regularizer=regularizers.l2(self.l2),
        )(activation_1)
        activation_2 = Activation(activation=self.activation)(cnn_layer_2)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(activation_2)
        flattener = Flatten()(maxpool_1)
        dropout_1 = Dropout(self.dropout_rate)(flattener)
        layer_3 = Dense(
            units=self.num_hidden[0],
            kernel_regularizer=regularizers.l2(self.l2),
            kernel_initializer=self.initializer,
        )(dropout_1)
        activation_3 = Activation(activation=self.activation)(layer_3)
        dropout_2 = Dropout(self.dropout_rate)(activation_3)
        layer_4 = Dense(
            units=self.num_hidden[1],
            kernel_regularizer=regularizers.l2(self.l2),
            kernel_initializer=self.initializer,
        )(dropout_2)
        activation_4 = Activation(activation=self.activation)(layer_4)
        layer_5 = Dense(units=self._labels_shape["output"])(activation_4)
        output = Activation(activation=self._last_layer_activation, name="output")(
            layer_5
        )

        model = Model(inputs=input_tensor, outputs=output)

        return model


class StarNet2017(CNNBase):
    """
    To create StarNet, S. Fabbro et al. (2017) arXiv:1709.09182. astroNN implemented the exact architecture with
    default parameter same as StarNet paper

    :History: 2017-Dec-23 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self):
        super().__init__()

        self.name = "StarNet (arXiv:1709.09182)"
        self._implementation_version = "1.0"
        self.initializer = "he_normal"
        self.activation = "relu"
        self.num_filters = [4, 16]
        self.filter_len = 8
        self.pool_length = 4
        self.num_hidden = [256, 128]
        self.max_epochs = 30
        self.lr = 0.0007
        self.l2 = 0.0
        self.reduce_lr_epsilon = 0.00005
        self.reduce_lr_min = 0.00008
        self.reduce_lr_patience = 2
        self.early_stopping_min_delta = 0.0001
        self.early_stopping_patience = 4

        self.input_norm_mode = 3

        self.task = "regression"

        self.targetname = ["teff", "logg", "Fe"]

    def model(self):
        input_tensor = Input(shape=self._input_shape["input"], name="input")
        cnn_layer_1 = Conv1D(
            kernel_initializer=self.initializer,
            activation=self.activation,
            padding="same",
            filters=self.num_filters[0],
            kernel_size=self.filter_len,
        )(input_tensor)
        cnn_layer_2 = Conv1D(
            kernel_initializer=self.initializer,
            activation=self.activation,
            padding="same",
            filters=self.num_filters[1],
            kernel_size=self.filter_len,
        )(cnn_layer_1)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(cnn_layer_2)
        flattener = Flatten()(maxpool_1)
        layer_3 = Dense(
            units=self.num_hidden[0],
            kernel_initializer=self.initializer,
            activation=self.activation,
        )(flattener)
        layer_4 = Dense(
            units=self.num_hidden[1],
            kernel_initializer=self.initializer,
            activation=self.activation,
        )(layer_3)
        layer_out = Dense(
            units=self._labels_shape["output"],
            kernel_initializer=self.initializer,
            activation=self._last_layer_activation,
            name="output",
        )(layer_4)
        model = Model(inputs=input_tensor, outputs=layer_out)

        return model


# noinspection PyCallingNonCallable
class ApogeeCVAE(ConvVAEBase):
    """
    Class for Convolutional Autoencoder Neural Network for stellar spectra analysis

    :History: 2017-Dec-21 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self):
        super().__init__()

        self._implementation_version = "1.0"
        self.batch_size = 64
        self.initializer = "he_normal"
        self.activation = "relu"
        self.optimizer = None
        self.num_filters = [2, 4]
        self.filter_len = 8
        self.pool_length = 4
        self.num_hidden = [128, 64]
        self.latent_dim = 2
        self.max_epochs = 100
        self.lr = 0.0005
        self.reduce_lr_epsilon = 0.0005
        self.reduce_lr_min = 0.0000000001
        self.reduce_lr_patience = 4
        self.epsilon_std = 1.0
        self.task = "regression"
        self.keras_encoder = None
        self.keras_vae = None
        self.l1 = 1e-5
        self.l2 = 1e-5
        self.dropout_rate = 0.1
        self._last_layer_activation = "linear"
        self.targetname = "Spectra Reconstruction"

        self.input_norm_mode = "2"
        self.labels_norm_mode = "2"

    def model(self):
        input_tensor = Input(shape=self._input_shape["input"], name="input")
        cnn_layer_1 = Conv1D(
            kernel_initializer=self.initializer,
            activation=self.activation,
            padding="same",
            filters=self.num_filters[0],
            kernel_size=self.filter_len,
            kernel_regularizer=regularizers.l2(self.l2),
        )(input_tensor)
        dropout_1 = Dropout(self.dropout_rate)(cnn_layer_1)
        cnn_layer_2 = Conv1D(
            kernel_initializer=self.initializer,
            activation=self.activation,
            padding="same",
            filters=self.num_filters[1],
            kernel_size=self.filter_len,
            kernel_regularizer=regularizers.l2(self.l2),
        )(dropout_1)
        dropout_2 = Dropout(self.dropout_rate)(cnn_layer_2)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(dropout_2)
        flattener = Flatten()(maxpool_1)
        layer_4 = Dense(
            units=self.num_hidden[0],
            kernel_regularizer=regularizers.l1(self.l1),
            kernel_initializer=self.initializer,
            activation=self.activation,
        )(flattener)
        dropout_3 = Dropout(self.dropout_rate)(layer_4)
        layer_5 = Dense(
            units=self.num_hidden[1],
            kernel_regularizer=regularizers.l1(self.l1),
            kernel_initializer=self.initializer,
            activation=self.activation,
        )(dropout_3)
        dropout_4 = Dropout(self.dropout_rate)(layer_5)
        z_mu = Dense(
            units=self.latent_dim,
            activation="linear",
            name="mean_output",
            kernel_initializer=self.initializer,
            kernel_regularizer=regularizers.l1(self.l1),
        )(dropout_4)
        z_log_var = Dense(
            units=self.latent_dim,
            activation="linear",
            name="sigma_output",
            kernel_initializer=self.initializer,
            kernel_regularizer=regularizers.l1(self.l1),
        )(dropout_4)

        z = VAESampling()([z_mu, z_log_var])

        decoder = Sequential(name="output")
        decoder.add(
            Dense(
                units=self.num_hidden[1],
                kernel_regularizer=regularizers.l1(self.l1),
                kernel_initializer=self.initializer,
                activation=self.activation,
                input_dim=self.latent_dim,
            )
        )
        decoder.add(Dropout(self.dropout_rate))
        decoder.add(
            Dense(
                units=self._input_shape["input"][0] * self.num_filters[1],
                kernel_regularizer=regularizers.l2(self.l2),
                kernel_initializer=self.initializer,
                activation=self.activation,
            )
        )
        decoder.add(Dropout(self.dropout_rate))
        output_shape = (
            self.batch_size,
            self._input_shape["input"][0],
            self.num_filters[1],
        )
        decoder.add(Reshape(output_shape[1:]))
        decoder.add(
            Conv1D(
                kernel_initializer=self.initializer,
                activation=self.activation,
                padding="same",
                filters=self.num_filters[1],
                kernel_size=self.filter_len,
                kernel_regularizer=regularizers.l2(self.l2),
            )
        )
        decoder.add(Dropout(self.dropout_rate))
        decoder.add(
            Conv1D(
                kernel_initializer=self.initializer,
                activation=self.activation,
                padding="same",
                filters=self.num_filters[0],
                kernel_size=self.filter_len,
                kernel_regularizer=regularizers.l2(self.l2),
            )
        )
        decoder.add(
            Conv1D(
                kernel_initializer=self.initializer,
                activation=self._last_layer_activation,
                padding="same",
                filters=1,
                kernel_size=self.filter_len,
                name="output",
            )
        )

        x_pred = decoder(z)
        # vae = Model(inputs=[input_tensor], outputs=[x_pred])
        encoder = Model(inputs=[input_tensor], outputs=[z_mu, z_log_var, z])

        return encoder, decoder


class DeNormAdd(tfk.layers.Layer):
    """
    Just a layer to work around `TypeError: can"t pickle _thread.lock objects` issue when saving this particular model

    For denormalizing
    """

    def __init__(self, norm, name=None, **kwargs):
        self.norm = norm
        self.supports_masking = True
        if not name:
            prefix = self.__class__.__name__
            name = prefix + "_" + str(tfk.backend.get_uid(prefix))
        super().__init__(name=name, **kwargs)

    def call(self, inputs, training=None):
        return tf.add(inputs, self.norm)

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {"norm": self.norm}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}


# noinspection PyCallingNonCallable
class ApogeeDR14GaiaDR2BCNN(BayesianCNNBase):
    """
    Class for Bayesian convolutional neural network for APOGEE DR14 Gaia DR2

    :History: 2018-Nov-06 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, lr=0.001, dropout_rate=0.3):
        super().__init__()

        self._implementation_version = "1.0"
        self.initializer = RandomNormal(mean=0.0, stddev=0.05)
        self.activation = "relu"
        self.num_filters = [2, 4]
        self.filter_len = 8
        self.pool_length = 4
        self.num_hidden = [162, 64, 32, 16]
        self.max_epochs = 100
        self.lr = lr
        self.reduce_lr_epsilon = 0.00005

        self.reduce_lr_min = 1e-8
        self.reduce_lr_patience = 2
        self.l2 = 5e-9
        self.dropout_rate = dropout_rate

        self.input_norm_mode = 3

        self.task = "regression"

        self.targetname = ["Ks-band fakemag"]

    def magmask(self):
        magmask = np.zeros(self._input_shape["input"][0], dtype=bool)
        magmask[7514] = True  # mask to extract extinction correction apparent magnitude
        return magmask

    def specmask(self):
        specmask = np.zeros(self._input_shape["input"][0], dtype=bool)
        specmask[
            :7514
        ] = True  # mask to extract extinction correction apparent magnitude
        return specmask

    def gaia_aux_mask(self):
        gaia_aux = np.zeros(self._input_shape["input"][0], dtype=bool)
        gaia_aux[7515:] = True  # mask to extract data for gaia offset
        return gaia_aux

    def model(self):
        input_tensor = Input(
            shape=self._input_shape["input"], name="input"
        )  # training data
        labels_err_tensor = Input(
            shape=(self._labels_shape["output"],), name="labels_err"
        )

        # extract spectra from input data and expand_dims for convolution
        spectra = Lambda(lambda x: tf.expand_dims(x, axis=-1))(
            BoolMask(self.specmask())(Flatten()(input_tensor))
        )

        # value to denorm magnitude
        app_mag = BoolMask(self.magmask())(Flatten()(input_tensor))
        # tf.convert_to_tensor(self.input_mean[self.magmask()])
        denorm_mag = DeNormAdd(np.array(self.input_mean["input"][self.magmask()]))(
            app_mag
        )
        inv_pow_mag = Lambda(lambda mag: tf.pow(10.0, tf.multiply(-0.2, mag)))(
            denorm_mag
        )

        # data to infer Gia DR2 offset
        # ========================== Offset Calibration Model ========================== #
        gaia_aux_data = BoolMask(self.gaia_aux_mask())(Flatten()(input_tensor))
        gaia_aux_hidden = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[2],
                kernel_regularizer=regularizers.l2(self.l2),
                kernel_initializer=self.initializer,
                activation="tanh",
            )(gaia_aux_data)
        )
        gaia_aux_hidden2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            Dense(
                units=self.num_hidden[3],
                kernel_regularizer=regularizers.l2(self.l2),
                kernel_initializer=self.initializer,
                activation="tanh",
            )(gaia_aux_hidden)
        )
        offset = Dense(
            units=1,
            kernel_initializer=self.initializer,
            activation="tanh",
            name="offset_output",
        )(gaia_aux_hidden2)
        # ========================== Offset Calibration Model ========================== #

        # good old NN takes spectra and output fakemag
        # ========================== Spectro-Luminosity Model ========================== #
        cnn_layer_1 = Conv1D(
            kernel_initializer=self.initializer,
            padding="same",
            filters=self.num_filters[0],
            kernel_size=self.filter_len,
            kernel_regularizer=regularizers.l2(self.l2),
        )(spectra)
        activation_1 = Activation(activation=self.activation)(cnn_layer_1)
        dropout_1 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            activation_1
        )
        cnn_layer_2 = Conv1D(
            kernel_initializer=self.initializer,
            padding="same",
            filters=self.num_filters[1],
            kernel_size=self.filter_len,
            kernel_regularizer=regularizers.l2(self.l2),
        )(dropout_1)
        activation_2 = Activation(activation=self.activation)(cnn_layer_2)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(activation_2)
        flattener = Flatten()(maxpool_1)
        dropout_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            flattener
        )
        layer_3 = Dense(
            units=self.num_hidden[0],
            kernel_regularizer=regularizers.l2(self.l2),
            kernel_initializer=self.initializer,
        )(dropout_2)
        activation_3 = Activation(activation=self.activation)(layer_3)
        dropout_3 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            activation_3
        )
        layer_4 = Dense(
            units=self.num_hidden[1],
            kernel_regularizer=regularizers.l2(self.l2),
            kernel_initializer=self.initializer,
        )(dropout_3)
        activation_4 = Activation(activation=self.activation)(layer_4)
        fakemag_output = Dense(
            units=self._labels_shape["output"],
            activation="softplus",
            name="fakemag_output",
        )(activation_4)
        fakemag_variance_output = Dense(
            units=self._labels_shape["output"],
            activation="linear",
            name="fakemag_variance_output",
        )(activation_4)
        # ========================== Spectro-Luminosity Model ========================== #

        # multiply a pre-determined de-normalization factor, such that fakemag std approx. 1 for training set
        # it does not really matter as NN will adapt to whatever value this is
        _fakemag_denorm = Lambda(lambda x: tf.multiply(x, 73.85))(fakemag_output)
        _fakemag_var_denorm = Lambda(lambda x: tf.add(x, tf.math.log(73.85)))(
            fakemag_variance_output
        )
        _fakemag_parallax = Multiply()([_fakemag_denorm, inv_pow_mag])

        # output parallax
        output = Add(name="output")([_fakemag_parallax, offset])
        variance_output = Lambda(
            lambda x: tf.math.log(
                tf.abs(tf.multiply(x[2], tf.divide(tf.exp(x[0]), x[1])))
            ),
            name="variance_output",
        )([fakemag_variance_output, fakemag_output, _fakemag_parallax])

        model = Model(
            inputs=[input_tensor, labels_err_tensor], outputs=[output, variance_output]
        )
        # new astroNN high performance dropout variational inference on GPU expects single output
        # while training with parallax, we want testing output fakemag
        model_prediction = Model(
            inputs=[input_tensor],
            outputs=concatenate([_fakemag_denorm, _fakemag_var_denorm]),
        )

        variance_loss = mse_var_wrapper(output, labels_err_tensor)
        output_loss = mse_lin_wrapper(variance_output, labels_err_tensor)

        return model, model_prediction, output_loss, variance_loss


class ApogeeKplerEchelle(CNNBase):
    """
    Class for Convolutional Neural Network for Echelle Diagram

    :History: 2020-Apr-06 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, lr=0.002):
        super().__init__()

        self._implementation_version = "1.0"
        self.initializer = "glorot_uniform"
        self.activation = "tanh"
        self.num_filters = [2, 4]
        self.filter_len = [8, 8]
        self.pool_length = 4
        self.num_hidden = [64, 32]
        self.max_epochs = 40
        self.lr = lr
        self.reduce_lr_epsilon = 0.00005

        self.reduce_lr_min = 1e-8
        self.reduce_lr_patience = 2
        self.l2 = 0.0
        self.dropout_rate = 0.1

        self.input_norm_mode = {"input": 255, "aux": 2}
        self.labels_norm_mode = 2

        self.task = "regression"
        self.targetname = []

    def model(self):
        input_tensor = Input(shape=self._input_shape["input"], name="input")
        aux_tensor = Input(shape=self._input_shape["aux"], name="aux")
        aux_flatten = Flatten()(aux_tensor)

        cnn_layer_1 = Conv2D(
            kernel_initializer=self.initializer,
            padding="valid",
            filters=self.num_filters[0],
            kernel_size=self.filter_len,
        )(input_tensor)
        activation_1 = Activation(activation=self.activation)(cnn_layer_1)
        cnn_layer_2 = Conv2D(
            kernel_initializer=self.initializer,
            padding="valid",
            filters=self.num_filters[1],
            kernel_size=self.filter_len,
        )(activation_1)
        activation_2 = Activation(activation=self.activation)(cnn_layer_2)
        maxpool_1 = MaxPooling2D(pool_size=self.pool_length)(activation_2)
        flattener = Flatten()(maxpool_1)
        dropout_1 = Dropout(self.dropout_rate)(flattener)
        layer_3 = Dense(
            units=self.num_hidden[0],
            kernel_regularizer=regularizers.l2(self.l2),
            kernel_initializer=self.initializer,
        )(dropout_1)
        activation_3 = Activation(activation=self.activation)(layer_3)
        dropout_2 = Dropout(self.dropout_rate)(activation_3)
        layer_4 = Dense(
            units=self.num_hidden[1],
            kernel_regularizer=regularizers.l2(self.l2),
            kernel_initializer=self.initializer,
        )(concatenate([dropout_2, aux_flatten]))
        activation_4 = Activation(activation=self.activation)(layer_4)
        layer_5 = Dense(
            units=self._labels_shape["output"],
            kernel_regularizer=regularizers.l2(self.l2),
            kernel_initializer=self.initializer,
        )(activation_4)
        output = Activation(activation=self._last_layer_activation, name="output")(
            layer_5
        )

        model = Model(inputs=[input_tensor, aux_tensor], outputs=[output])

        return model


class ApogeeBCNNaux(BayesianCNNBase):
    """
    Class for Bayesian convolutional neural network for APOGEE with auxiliary data

    :History: 2022-May-09 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, lr=0.001, dropout_rate=0.3):
        super().__init__()

        self._implementation_version = "1.0"
        self.initializer = RandomNormal(mean=0.0, stddev=0.05)
        self.activation = "relu"
        self.num_filters = [2, 4]
        self.filter_len = 8
        self.pool_length = 4
        self.num_hidden = [162, 64, 32, 16]
        self.max_epochs = 100
        self.lr = lr
        self.reduce_lr_epsilon = 0.00005

        self.reduce_lr_min = 1e-8
        self.reduce_lr_patience = 2
        self.l2 = 5e-9
        self.dropout_rate = dropout_rate

        self.input_norm_mode = 2
        self.aux_length = 2

        self.task = "regression"

        self.targetname = ["Mass"]

    def specmask(self):
        specmask = np.zeros(self._input_shape["input"][0], dtype=bool)
        specmask[
            : -self.aux_length
        ] = True  # mask to extract extinction correction apparent magnitude
        return specmask

    def aux_mask(self):
        # teff and fe_h
        aux = np.zeros(self._input_shape["input"][0], dtype=bool)
        aux[-self.aux_length :] = True  # mask to extract data
        return aux

    def model(self):
        input_tensor = Input(
            shape=self._input_shape["input"], name="input"
        )  # training data
        labels_err_tensor = Input(
            shape=(self._labels_shape["output"],), name="labels_err"
        )

        # extract spectra from input data and expand_dims for convolution
        spectra = Lambda(lambda x: tf.expand_dims(x, axis=-1))(
            BoolMask(self.specmask())(Flatten()(input_tensor))
        )

        # data to infer Gia DR2 offset
        # ========================== additional data ========================== #
        aux_data = BoolMask(self.aux_mask())(Flatten()(input_tensor))

        # good old NN takes spectra and output fakemag
        # ========================== Main Model ========================== #
        cnn_layer_1 = Conv1D(
            kernel_initializer=self.initializer,
            padding="same",
            filters=self.num_filters[0],
            kernel_size=self.filter_len,
            kernel_regularizer=regularizers.l2(self.l2),
        )(spectra)
        activation_1 = Activation(activation=self.activation)(cnn_layer_1)
        dropout_1 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            activation_1
        )
        cnn_layer_2 = Conv1D(
            kernel_initializer=self.initializer,
            padding="same",
            filters=self.num_filters[1],
            kernel_size=self.filter_len,
            kernel_regularizer=regularizers.l2(self.l2),
        )(dropout_1)
        activation_2 = Activation(activation=self.activation)(cnn_layer_2)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(activation_2)
        flattener = Flatten()(maxpool_1)
        dropout_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            flattener
        )
        layer_3 = Dense(
            units=self.num_hidden[0],
            kernel_regularizer=regularizers.l2(self.l2),
            kernel_initializer=self.initializer,
        )(concatenate([dropout_2, aux_data]))
        activation_3 = Activation(activation=self.activation)(layer_3)
        dropout_3 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            activation_3
        )
        layer_4 = Dense(
            units=self.num_hidden[1],
            kernel_regularizer=regularizers.l2(self.l2),
            kernel_initializer=self.initializer,
        )(dropout_3)
        activation_4 = Activation(activation=self.activation)(layer_4)
        output = Dense(
            units=self._labels_shape["output"], activation="linear", name="output"
        )(activation_4)
        variance_output = Dense(
            units=self._labels_shape["output"],
            activation="linear",
            name="variance_output",
        )(activation_4)
        # ========================== Main Model ========================== #

        model = Model(
            inputs=[input_tensor, labels_err_tensor], outputs=[output, variance_output]
        )
        model_prediction = Model(
            inputs=[input_tensor], outputs=concatenate([output, variance_output])
        )

        variance_loss = mse_var_wrapper(output, labels_err_tensor)
        output_loss = mse_lin_wrapper(variance_output, labels_err_tensor)

        return model, model_prediction, output_loss, variance_loss


class ApokascEncoderDecoder(ConvVAEBase):
    def __init__(self, lr=0.0005, dropout_rate=0.0):
        super().__init__()

        self._implementation_version = "1.0"
        self.batch_size = 128
        self.initializer = "glorot_uniform"
        self.activation = "relu"
        self.num_filters = [32, 64, 16, 16]
        self.filter_len = [8, 32]
        self.pool_length = 2
        self.num_hidden = [16, 16]
        self.latent_dim = 5
        self.max_epochs = 100
        self.lr = lr
        self.optimizer = tfk.optimizers.Adam(learning_rate=self.lr)
        self.reduce_lr_epsilon = 0.00005
        self.reduce_lr_min = 0.0000000001
        self.reduce_lr_patience = 6
        self.epsilon_std = 1.0
        self.task = "regression"
        self.keras_encoder = None
        self.keras_vae = None
        self.l1 = 1e-5
        self.l2 = 1e-5
        self.dropout_rate = dropout_rate
        self._last_layer_activation = "linear"
        self.targetname = "PSD"
        self.nn_output_internal = -1

        self.input_norm_mode = "2"
        self.labels_norm_mode = "0"

    def model(self):
        self.nn_output_internal = self._labels_shape["output"] // 4
        encoder_inputs = Input(shape=self._input_shape["input"], name="input")
        x = Conv1D(
            self.num_filters[0],
            self.filter_len[0],
            activation=self.activation,
            strides=2,
            padding="same",
            kernel_initializer=self.initializer,
            kernel_regularizer=regularizers.l2(self.l2),
        )(encoder_inputs)
        x = Dropout(self.dropout_rate)(x)
        x = Conv1D(
            self.num_filters[1],
            self.filter_len[0],
            activation=self.activation,
            strides=2,
            padding="same",
            kernel_initializer=self.initializer,
            kernel_regularizer=regularizers.l2(self.l2),
        )(x)
        x = MaxPooling1D(pool_size=self.pool_length)(x)
        x = Dropout(self.dropout_rate)(x)
        x = Flatten()(x)
        x = Dense(
            self.num_hidden[0],
            activation="tanh",
            kernel_initializer=self.initializer,
            kernel_regularizer=regularizers.l2(self.l2),
        )(x)
        z_mean = Dense(
            self.latent_dim,
            name="z_mean",
            kernel_initializer=self.initializer,
            kernel_regularizer=regularizers.l2(self.l2),
        )(x)
        z_log_var = Dense(
            self.latent_dim,
            name="z_log_var",
            kernel_initializer=self.initializer,
            kernel_regularizer=regularizers.l2(self.l2),
        )(x)
        z = VAESampling()([z_mean, z_log_var])
        encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        latent_inputs = Input(shape=(self.latent_dim,), name="decoder_input")
        x = Dense(
            self.nn_output_internal * self.num_hidden[1],
            activation=self.activation,
            kernel_initializer=self.initializer,
            kernel_regularizer=regularizers.l2(self.l2),
        )(latent_inputs)
        x = Dropout(self.dropout_rate)(x)
        x = Reshape((self.nn_output_internal, self.num_hidden[1]))(x)
        x = Conv1DTranspose(
            self.num_filters[2],
            self.filter_len[1],
            activation=self.activation,
            strides=2,
            padding="same",
            kernel_initializer=self.initializer,
            kernel_regularizer=regularizers.l2(self.l2),
        )(x)
        x = Dropout(self.dropout_rate)(x)
        x = Conv1DTranspose(
            self.num_filters[3],
            self.filter_len[1],
            activation=self.activation,
            strides=2,
            padding="same",
            kernel_initializer=self.initializer,
            kernel_regularizer=regularizers.l2(self.l2),
        )(x)
        x = Dropout(self.dropout_rate)(x)
        decoder_outputs = Conv1DTranspose(
            1,
            self.filter_len[1],
            padding="same",
            kernel_initializer=self.initializer,
            kernel_regularizer=regularizers.l2(self.l2),
            name="output",
        )(x)
        decoder = Model(latent_inputs, decoder_outputs, name="output")
        return encoder, decoder
