# ---------------------------------------------------------#
#   astroNN.models.CIFAR10_CNN: Contain CNN Model
# ---------------------------------------------------------#
from tensorflow import keras as tfk

from astroNN.models.base_cnn import CNNBase
from astroNN.nn.layers import MCDropout

regularizers = tfk.regularizers
MaxPooling2D, Conv2D, Dense, Flatten, Activation, Input = tfk.layers.MaxPooling2D, tfk.layers.Conv2D, \
                                                          tfk.layers.Dense, tfk.layers.Flatten, \
                                                          tfk.layers.Activation, tfk.layers.Input
max_norm = tfk.constraints.max_norm
Model = tfk.models.Model


class CustomModel_Test(CNNBase):
    """
    NAME:
        CustomModel_Test
    PURPOSE:
        To create a custom Convolutional Neural Network model for testing
    HISTORY:
        2018-Mar-11 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, lr=0.005):
        """
        NAME:
            model
        PURPOSE:
            To create a custom Convolutional Neural Network model for testing
        INPUT:
        OUTPUT:
        HISTORY:
        2018-Mar-11 - Written - Henry Leung (University of Toronto)
        """
        super().__init__()

        self._implementation_version = '1.0'
        self.initializer = 'he_normal'
        self.activation = 'relu'
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

        self.task = 'classification'
        self.targetname = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.input_norm_mode = 255
        self.labels_norm_mode = 0

    def __call__(self):
        return self

    def model(self):
        input_tensor = Input(shape=self._input_shape, name='input')
        cnn_layer_1 = Conv2D(kernel_initializer=self.initializer, padding="same", filters=self.num_filters[0],
                             kernel_size=self.filter_len, kernel_regularizer=regularizers.l2(self.l2))(input_tensor)
        activation_1 = Activation(activation=self.activation)(cnn_layer_1)
        cnn_layer_2 = Conv2D(kernel_initializer=self.initializer, padding="same", filters=self.num_filters[1],
                             kernel_size=self.filter_len, kernel_regularizer=regularizers.l2(self.l2))(activation_1)
        activation_2 = Activation(activation=self.activation)(cnn_layer_2)
        maxpool_1 = MaxPooling2D(pool_size=self.pool_length)(activation_2)
        flattener = Flatten()(maxpool_1)
        dropout_1 = MCDropout(0.2, disable=True)(flattener)
        layer_3 = Dense(units=self.num_hidden[0], kernel_regularizer=regularizers.l2(self.l2),
                        kernel_initializer=self.initializer)(dropout_1)
        activation_3 = Activation(activation=self.activation)(layer_3)
        dropout_2 = MCDropout(0.2, disable=True)(activation_3)
        layer_4 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l2(self.l2),
                        kernel_initializer=self.initializer, kernel_constraint=max_norm(2))(dropout_2)
        activation_4 = Activation(activation=self.activation)(layer_4)
        layer_5 = Dense(units=self._labels_shape)(activation_4)
        output = Activation(activation=self._last_layer_activation, name='output')(layer_5)

        model = Model(inputs=input_tensor, outputs=output)

        return model
