# ---------------------------------------------------------#
#   astroNN.models.CNN: Contain CNN Model
# ---------------------------------------------------------#
import os

from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.layers import MaxPooling1D, Conv1D, Dense, Dropout, Flatten, Activation
from keras.models import Model, Input

from astroNN.apogee.plotting import ASPCAP_plots
from astroNN.models.CNNBase import CNNBase


class APOGEE_CNN(CNNBase, ASPCAP_plots):
    """
    NAME:
        CNN
    PURPOSE:
        To create Convolutional Neural Network model
    HISTORY:
        2017-Dec-21 - Written - Henry Leung (University of Toronto)
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
            2017-Dec-21 - Written - Henry Leung (University of Toronto)
        """
        super(APOGEE_CNN, self).__init__()

        self._model_identifier = 'APOGEE_CNN'
        self._implementation_version = '1.0'
        self.batch_size = 64
        self.initializer = 'he_normal'
        self.activation = 'relu'
        self.num_filters = [2, 4]
        self.filter_len = (8)
        self.pool_length = (4)
        self.num_hidden = [196, 96]
        self.max_epochs = 100
        self.lr = lr
        self.reduce_lr_epsilon = 0.00005

        self.reduce_lr_min = 1e-8
        self.reduce_lr_patience = 2
        self.target = 'all'
        self.l2 = 1e-8

        self.task = 'regression'
        self.targetname = ['teff', 'logg', 'M', 'alpha', 'C', 'C1', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'K',
                           'Ca', 'Ti', 'Ti2', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'fakemag']

    def model(self):
        input_tensor = Input(shape=self.input_shape)
        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, padding="same", filters=self.num_filters[0],
                             kernel_size=self.filter_len, kernel_regularizer=regularizers.l2(self.l2))(input_tensor)
        activation_1 = Activation(activation=self.activation)(cnn_layer_1)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, padding="same", filters=self.num_filters[1],
                             kernel_size=self.filter_len, kernel_regularizer=regularizers.l2(self.l2))(activation_1)
        activation_2 = Activation(activation=self.activation)(cnn_layer_2)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(activation_2)
        flattener = Flatten()(maxpool_1)
        dropout_1 = Dropout(0.05)(flattener)
        layer_3 = Dense(units=self.num_hidden[0], kernel_regularizer=regularizers.l2(self.l2),
                        kernel_initializer=self.initializer)(dropout_1)
        activation_3 = Activation(activation=self.activation)(layer_3)
        dropout_2 = Dropout(0.05)(activation_3)
        layer_4 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l2(self.l2),
                        kernel_initializer=self.initializer)(dropout_2)
        activation_4 = Activation(activation=self.activation)(layer_4)
        layer_5 = Dense(units=self.labels_shape)(activation_4)
        output = Activation(activation=self._last_layer_activation)(layer_5)

        model = Model(inputs=input_tensor, outputs=output)

        return model

    def train(self, input_data, labels):
        # Call the checklist to create astroNN folder and save parameters
        self.pre_training_checklist_child(input_data, labels)

        csv_logger = CSVLogger(self.fullfilepath + 'log.csv', append=True, separator=',')

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=self.reduce_lr_epsilon,
                                      patience=self.reduce_lr_patience, min_lr=self.reduce_lr_min, mode='min',
                                      verbose=2)

        self.keras_model.fit_generator(generator=self.training_generator,
                                       steps_per_epoch=self.num_train // self.batch_size,
                                       epochs=self.max_epochs, verbose=2, workers=os.cpu_count(),
                                       callbacks=[reduce_lr, csv_logger])

        # Call the post training checklist to save parameters
        self.post_training_checklist_child()

        return None
