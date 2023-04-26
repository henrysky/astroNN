# ---------------------------------------------------------#
#   astroNN.models.misc_models: Contain Misc. Models
# ---------------------------------------------------------#
from tensorflow import keras as tfk

from astroNN.models.base_bayesian_cnn import BayesianCNNBase
from astroNN.models.base_cnn import CNNBase
from astroNN.nn.layers import MCDropout, PolyFit
from astroNN.nn.losses import (
    bayesian_binary_crossentropy_wrapper,
    bayesian_binary_crossentropy_var_wrapper,
)
from astroNN.nn.losses import (
    bayesian_categorical_crossentropy_wrapper,
    bayesian_categorical_crossentropy_var_wrapper,
)

regularizers = tfk.regularizers

Dense = tfk.layers.Dense
Input = tfk.layers.Input
Conv2D = tfk.layers.Conv2D
Dropout = tfk.layers.Dropout
Flatten = tfk.layers.Flatten
Activation = tfk.layers.Activation
concatenate = tfk.layers.concatenate
MaxPooling2D = tfk.layers.MaxPooling2D

Model = tfk.models.Model
MaxNorm = tfk.constraints.MaxNorm


class Cifar10CNN(CNNBase):
    """
    NAME:
        Cifar10CNN
    PURPOSE:
        To create Convolutional Neural Network model for Cifar10 for the purpose of demo
    HISTORY:
        2018-Jan-11 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, lr=0.005):
        """
        NAME:
            model
        PURPOSE:
            To create Convolutional Neural Network model
        INPUT:
        OUTPUT:
        HISTORY:
            2018-Jan-11 - Written - Henry Leung (University of Toronto)
        """
        super().__init__()

        self._implementation_version = "1.0"
        self.initializer = "he_normal"
        self.activation = "relu"
        self.num_filters = [8, 16]
        self.filter_len = (3, 3)
        self.pool_length = (4, 4)
        self.num_hidden = [256, 128]
        self.max_epochs = 30
        self.lr = lr
        self.reduce_lr_epsilon = 0.00005

        self.reduce_lr_min = 1e-8
        self.reduce_lr_patience = 1
        self.l2 = 1e-4
        self.dropout_rate = 0.1

        self.task = "classification"
        self.targetname = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        self.input_norm_mode = 255
        self.labels_norm_mode = 0

    def model(self):
        input_tensor = Input(shape=self._input_shape["input"], name="input")
        cnn_layer_1 = Conv2D(
            kernel_initializer=self.initializer,
            padding="same",
            filters=self.num_filters[0],
            kernel_size=self.filter_len,
            kernel_regularizer=regularizers.l2(self.l2),
        )(input_tensor)
        activation_1 = Activation(activation=self.activation)(cnn_layer_1)
        cnn_layer_2 = Conv2D(
            kernel_initializer=self.initializer,
            padding="same",
            filters=self.num_filters[1],
            kernel_size=self.filter_len,
            kernel_regularizer=regularizers.l2(self.l2),
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
            kernel_constraint=MaxNorm(2),
        )(dropout_2)
        activation_4 = Activation(activation=self.activation)(layer_4)
        layer_5 = Dense(units=self._labels_shape["output"])(activation_4)
        output = Activation(activation=self._last_layer_activation, name="output")(
            layer_5
        )

        model = Model(inputs=input_tensor, outputs=output)

        return model


# noinspection PyCallingNonCallable
class MNIST_BCNN(BayesianCNNBase):
    """
    NAME:
        MNIST_BCNN
    PURPOSE:
        To create Convolutional Neural Network model for Cifar10 for the purpose of demo
    HISTORY:
        2018-Jan-11 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, lr=0.005):
        """
        NAME:
            model
        PURPOSE:
            To create Convolutional Neural Network model
        INPUT:
        OUTPUT:
        HISTORY:
            2018-Jan-11 - Written - Henry Leung (University of Toronto)
        """
        super().__init__()

        self._implementation_version = "1.0"
        self.initializer = "he_normal"
        self.activation = "relu"
        self.num_filters = [8, 16]
        self.filter_len = (3, 3)
        self.pool_length = (4, 4)
        self.num_hidden = [256, 128]
        self.max_epochs = 30
        self.lr = lr
        self.reduce_lr_epsilon = 0.00005

        self.reduce_lr_min = 1e-8
        self.reduce_lr_patience = 1
        self.l2 = 1e-4
        self.dropout_rate = 0.1

        self.task = "classification"
        self.targetname = [
            "Zero",
            "One",
            "Two",
            "Three",
            "Four",
            "Five",
            "Six",
            "Seven",
            "Eight",
            "Nine",
        ]
        self.input_norm_mode = 255
        self.labels_norm_mode = 0

    def model(self):
        input_tensor = Input(shape=self._input_shape["input"], name="input")
        cnn_layer_1 = Conv2D(
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
        cnn_layer_2 = Conv2D(
            kernel_initializer=self.initializer,
            padding="same",
            filters=self.num_filters[1],
            kernel_size=self.filter_len,
            kernel_regularizer=regularizers.l2(self.l2),
        )(dropout_1)
        activation_2 = Activation(activation=self.activation)(cnn_layer_2)
        dropout_2 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            activation_2
        )
        maxpool_1 = MaxPooling2D(pool_size=self.pool_length)(dropout_2)
        flattener = Flatten()(maxpool_1)
        layer_3 = Dense(
            units=self.num_hidden[0],
            kernel_regularizer=regularizers.l2(self.l2),
            kernel_initializer=self.initializer,
        )(flattener)
        activation_3 = Activation(activation=self.activation)(layer_3)
        dropout_4 = MCDropout(self.dropout_rate, disable=self.disable_dropout)(
            activation_3
        )
        layer_4 = Dense(
            units=self.num_hidden[1],
            kernel_regularizer=regularizers.l2(self.l2),
            kernel_initializer=self.initializer,
            kernel_constraint=MaxNorm(2),
        )(dropout_4)
        activation_4 = Activation(activation=self.activation)(layer_4)
        output = Dense(
            units=self._labels_shape["output"], activation="linear", name="output"
        )(activation_4)
        output_activated = Activation(self._last_layer_activation)(output)
        variance_output = Dense(
            units=self._labels_shape["output"],
            activation="softplus",
            name="variance_output",
        )(activation_4)

        model = Model(inputs=[input_tensor], outputs=[output, variance_output])
        # new astroNN high performance dropout variational inference on GPU expects single output
        model_prediction = Model(
            inputs=[input_tensor],
            outputs=concatenate([output_activated, variance_output]),
        )

        if self.task == "classification":
            output_loss = bayesian_categorical_crossentropy_wrapper(variance_output)
            variance_loss = bayesian_categorical_crossentropy_var_wrapper(output)
        elif self.task == "binary_classification":
            output_loss = bayesian_binary_crossentropy_wrapper(variance_output)
            variance_loss = bayesian_binary_crossentropy_var_wrapper(output)
        else:
            raise RuntimeError(
                'Only "regression", "classification" and "binary_classification" are supported'
            )

        return model, model_prediction, output_loss, variance_loss


# noinspection PyCallingNonCallable
class SimplePolyNN(CNNBase):
    """
    Class for Neural Network for Gaia Polynomial fitting

    :History: 2018-Jul-23 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, lr=0.005, init_w=None, use_xbias=False):
        super().__init__()

        self._implementation_version = "1.0"
        self.max_epochs = 40
        self.lr = lr
        self.reduce_lr_epsilon = 0.00005
        self.num_hidden = 3  # equals degree of polynomial to fit

        self.reduce_lr_min = 1e-8
        self.reduce_lr_patience = 2

        self.input_norm_mode = 0
        self.labels_norm_mode = 0
        self.init_w = init_w
        self.use_xbias = use_xbias
        self.task = "regression"
        self.targetname = ["unbiased_parallax"]

    def model(self):
        input_tensor = Input(shape=self._input_shape, name="input")
        flattener = Flatten()(input_tensor)
        output = PolyFit(
            deg=self.num_hidden,
            output_units=self._labels_shape,
            use_xbias=self.use_xbias,
            name="output",
            init_w=self.init_w,
            kernel_regularizer=regularizers.l2(self.l2),
        )(flattener)

        model = Model(inputs=input_tensor, outputs=output)

        return model
