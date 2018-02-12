# ---------------------------------------------------------#
#   astroNN.models.BCNN: Contain BCNN Model
# ---------------------------------------------------------#
import os

from astroNN import MULTIPROCESS_FLAG
from astroNN.apogee.plotting import ASPCAP_plots
from astroNN.models.BayesianCNNBase import BayesianCNNBase
from astroNN.nn.losses import mse_lin_wrapper, mse_var_wrapper
from astroNN.nn.utilities.custom_layers import BayesianDropout, ErrorProp
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.layers import MaxPooling1D, Conv1D, Dense, Flatten, Activation
from keras.models import Model, Input


class Apogee_BCNN(BayesianCNNBase, ASPCAP_plots):
    """
    NAME:
        Apogee_BCNN
    PURPOSE:
        To create Bayesian Convolutional Neural Network model
    HISTORY:
        2017-Dec-21 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, lr=0.0005):
        """
        NAME:
            __init__
        PURPOSE:
            To create Bayesian Convolutional Neural Network model
        INPUT:
        OUTPUT:
        HISTORY:
            2017-Dec-21 - Written - Henry Leung (University of Toronto)
        """
        super(Apogee_BCNN, self).__init__()

        self._model_identifier = 'APOGEE_BCNN'
        self._implementation_version = '1.0'
        self.batch_size = 64
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
        self.dropout_rate = 0.3

        self.input_norm_mode = 3

        self.task = 'regression'

        self.targetname = ['teff', 'logg', 'M', 'alpha', 'C', 'C1', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'K',
                           'Ca', 'Ti', 'Ti2', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'fakemag']

    def model(self):
        input_tensor = Input(shape=self.input_shape, name='input')
        labels_err_tensor = Input(shape=(self.labels_shape,), name='labels_err')
        input_err_tensor = Input(shape=self.input_shape, name='input_err')

        input_with_err = ErrorProp(input_err_tensor)(input_tensor)

        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, padding="same", filters=self.num_filters[0],
                             kernel_size=self.filter_len, kernel_regularizer=regularizers.l2(self.l2))(input_with_err)
        activation_1 = Activation(activation=self.activation)(cnn_layer_1)
        dropout_1 = BayesianDropout(self.dropout_rate)(activation_1)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, padding="same", filters=self.num_filters[1],
                             kernel_size=self.filter_len, kernel_regularizer=regularizers.l2(self.l2))(dropout_1)
        activation_2 = Activation(activation=self.activation)(cnn_layer_2)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(activation_2)
        flattener = Flatten()(maxpool_1)
        dropout_2 = BayesianDropout(self.dropout_rate)(flattener)
        layer_3 = Dense(units=self.num_hidden[0], kernel_regularizer=regularizers.l2(self.l2),
                        kernel_initializer=self.initializer,
                        activation=self.activation)(dropout_2)
        activation_3 = Activation(activation=self.activation)(layer_3)
        dropout_3 = BayesianDropout(self.dropout_rate)(activation_3)
        layer_4 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l2(self.l2),
                        kernel_initializer=self.initializer,
                        activation=self.activation)(dropout_3)
        activation_4 = Activation(activation=self.activation)(layer_4)
        output = Dense(units=self.labels_shape, activation=self._last_layer_activation, name='output')(activation_4)
        variance_output = Dense(units=self.labels_shape, activation='linear', name='variance_output')(activation_4)

        model = Model(inputs=[input_tensor, labels_err_tensor, input_err_tensor], outputs=[output, variance_output])
        model_prediction = Model(inputs=[input_tensor, input_err_tensor], outputs=[output, variance_output])

        variance_loss = mse_var_wrapper(output, labels_err_tensor)
        output_loss = mse_lin_wrapper(variance_output, labels_err_tensor)

        return model, model_prediction, output_loss, variance_loss

    def train(self, input_data, labels, inputs_err, labels_err):
        # Call the checklist to create astroNN folder and save parameters
        self.pre_training_checklist_child(input_data, labels, labels_err)

        csv_logger = CSVLogger(self.fullfilepath + 'log.csv', append=True, separator=',')

        reduce_lr = ReduceLROnPlateau(monitor='val_output_loss', factor=0.5, epsilon=self.reduce_lr_epsilon,
                                      patience=self.reduce_lr_patience, min_lr=self.reduce_lr_min, mode='min',
                                      verbose=2)

        self.keras_model.fit_generator(generator=self.training_generator,
                                       steps_per_epoch=self.num_train // self.batch_size,
                                       validation_data=self.validation_generator,
                                       validation_steps=self.val_num // self.batch_size,
                                       epochs=self.max_epochs, verbose=2, workers=os.cpu_count(),
                                       callbacks=[reduce_lr, csv_logger], use_multiprocessing=MULTIPROCESS_FLAG)

        # Call the post training checklist to save parameters
        self.post_training_checklist_child()

        return None
