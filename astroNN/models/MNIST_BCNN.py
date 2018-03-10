# ---------------------------------------------------------#
#   astroNN.models.CIFAR10_CNN: Contain CNN Model
# ---------------------------------------------------------#
from astroNN.models.BayesianCNNBase import BayesianCNNBase
from astroNN.nn.layers import MCDropout
from astroNN.nn.losses import bayesian_crossentropy_wrapper
from astroNN import keras_import_manager

keras = keras_import_manager()
regularizers = keras.regularizers
MaxPooling2D, Conv2D, Dense, Flatten, Activation, Input = keras.layers.MaxPooling2D, keras.layers.Conv2D, \
                                                          keras.layers.Dense, keras.layers.Flatten, \
                                                          keras.layers.Activation, keras.layers.Input
max_norm = keras.constraints.max_norm
Model = keras.models.Model


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
        super(MNIST_BCNN, self).__init__()

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
        self.targetname = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
        self.input_norm_mode = 255
        self.labels_norm_mode = 0

    def __call__(self):
        return self

    def model(self):
        input_tensor = Input(shape=self.input_shape, name='input')
        cnn_layer_1 = Conv2D(kernel_initializer=self.initializer, padding="same", filters=self.num_filters[0],
                             kernel_size=self.filter_len, kernel_regularizer=regularizers.l2(self.l2))(input_tensor)
        activation_1 = Activation(activation=self.activation)(cnn_layer_1)
        dropout_1 = MCDropout(self.dropout_rate, disable=self.diable_dropout)(activation_1)
        cnn_layer_2 = Conv2D(kernel_initializer=self.initializer, padding="same", filters=self.num_filters[1],
                             kernel_size=self.filter_len, kernel_regularizer=regularizers.l2(self.l2))(dropout_1)
        activation_2 = Activation(activation=self.activation)(cnn_layer_2)
        dropout_2 = MCDropout(self.dropout_rate, disable=self.diable_dropout)(activation_2)
        maxpool_1 = MaxPooling2D(pool_size=self.pool_length)(dropout_2)
        flattener = Flatten()(maxpool_1)
        dropout_3 = MCDropout(self.dropout_rate, disable=self.diable_dropout)(flattener)
        layer_3 = Dense(units=self.num_hidden[0], kernel_regularizer=regularizers.l2(self.l2),
                        kernel_initializer=self.initializer)(dropout_3)
        activation_3 = Activation(activation=self.activation)(layer_3)
        dropout_4 = MCDropout(self.dropout_rate, disable=self.diable_dropout)(activation_3)
        layer_4 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l2(self.l2),
                        kernel_initializer=self.initializer, kernel_constraint=max_norm(2))(dropout_4)
        activation_4 = Activation(activation=self.activation)(layer_4)
        layer_5 = Dense(units=self.labels_shape)(activation_4)
        output = Activation(activation=self._last_layer_activation, name='output')(layer_5)
        mc_output = Activation(activation='linear', name='variance_output')(layer_5)

        model = Model(inputs=[input_tensor], outputs=[mc_output])
        model_prediction = Model(inputs=[input_tensor], outputs=[output])

        output_loss, variance_loss = bayesian_crossentropy_wrapper()

        return model, model_prediction, output_loss, variance_loss
