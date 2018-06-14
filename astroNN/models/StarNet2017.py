# ---------------------------------------------------------#
#   astroNN.models.StarNet2017: Contain starnet Model
# ---------------------------------------------------------#
from astroNN.apogee.plotting import ASPCAP_plots
from astroNN.config import keras_import_manager
from astroNN.models.CNNBase import CNNBase

keras = keras_import_manager()
regularizers = keras.regularizers
MaxPooling1D, Conv1D, Dense, Flatten, Activation, Input = keras.layers.MaxPooling1D, keras.layers.Conv1D, \
                                                          keras.layers.Dense, keras.layers.Flatten, \
                                                          keras.layers.Activation, keras.layers.Input
Model = keras.models.Model


class StarNet2017(CNNBase, ASPCAP_plots):
    """
    To create StarNet, S. Fabbro et al. (2017) arXiv:1709.09182. astroNN implemented the exact architecture with
    default parameter same as StarNet paper

    :History: 2017-Dec-23 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self):
        super().__init__()

        self.name = 'StarNet (arXiv:1709.09182)'
        self._implementation_version = '1.0'
        self.initializer = 'he_normal'
        self.activation = 'relu'
        self.num_filters = [4, 16]
        self.filter_len = 8
        self.pool_length = 4
        self.num_hidden = [256, 128]
        self.max_epochs = 30
        self.lr = 0.0007
        self.l2 = 0.
        self.reduce_lr_epsilon = 0.00005
        self.reduce_lr_min = 0.00008
        self.reduce_lr_patience = 2
        self.early_stopping_min_delta = 0.0001
        self.early_stopping_patience = 4

        self.input_norm_mode = 3

        self.task = 'regression'

        self.targetname = ['teff', 'logg', 'Fe']

    def model(self):
        input_tensor = Input(shape=self._input_shape, name='input')
        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0], kernel_size=self.filter_len)(input_tensor)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[1], kernel_size=self.filter_len)(cnn_layer_1)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(cnn_layer_2)
        flattener = Flatten()(maxpool_1)
        layer_3 = Dense(units=self.num_hidden[0], kernel_initializer=self.initializer, activation=self.activation)(
            flattener)
        layer_4 = Dense(units=self.num_hidden[1], kernel_initializer=self.initializer, activation=self.activation)(
            layer_3)
        layer_out = Dense(units=self._labels_shape, kernel_initializer=self.initializer,
                          activation=self._last_layer_activation, name='output')(layer_4)
        model = Model(inputs=input_tensor, outputs=layer_out)

        return model
