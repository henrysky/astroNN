# ---------------------------------------------------------#
#   astroNN.models.StarNet2017: Contain starnet Model
# ---------------------------------------------------------#
import os

import numpy as np
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.layers import MaxPooling1D, Conv1D, Dense, Flatten
from keras.models import Model, Input

from astroNN.models.NeuralNetBases import CNNBase
from astroNN.models.utilities.generator import DataGenerator
from astroNN.models.utilities.normalizer import Normalizer


class StarNet2017(CNNBase):
    """
    NAME:
        StarNet
    PURPOSE:
        To create StarNet, S. Fabbro et al. (2017) arXiv:1709.09182. astroNN implemented the exact architecture with
        default parameter same as StarNet paper
    HISTORY:
        2017-Dec-23 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self):
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
        super(StarNet2017, self).__init__()

        self.name = 'StarNet (arXiv:1709.09182)'
        self._model_type = 'CNN'
        self._model_identifier = 'StarNet2017'
        self._implementation_version = '1.0'
        self.batch_size = 64
        self.initializer = 'he_normal'
        self.activation = 'relu'
        self.num_filters = [4, 16]
        self.filter_length = 8
        self.pool_length = 4
        self.num_hidden = [256, 128]
        self.max_epochs = 30
        self.lr = 0.0007
        self.l2_penalty = 0.
        self.reduce_lr_epsilon = 0.00005
        self.reduce_lr_min = 0.00008
        self.reduce_lr_patience = 2
        self.early_stopping_min_delta = 0.0001
        self.early_stopping_patience = 4
        self.data_normalization = True
        self.target = ['teff', 'logg', 'Fe']

    def model(self):
        input_tensor = Input(shape=self.input_shape)
        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0], kernel_size=self.filter_length)(input_tensor)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[1], kernel_size=self.filter_length)(cnn_layer_1)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(cnn_layer_2)
        flattener = Flatten()(maxpool_1)
        layer_3 = Dense(units=self.num_hidden[0], kernel_initializer=self.initializer, activation=self.activation)(
            flattener)
        layer_4 = Dense(units=self.num_hidden[1], kernel_initializer=self.initializer, activation=self.activation)(
            layer_3)
        layer_out = Dense(units=self.labels_shape[0], kernel_initializer=self.initializer, activation=self.activation)(
            layer_4)
        model = Model(inputs=input_tensor, outputs=layer_out)

        return model

    def train(self, input_data, labels):
        if self.task == 'classification':
            raise RuntimeError('astroNN StarNet does not support classification task')

        # Call the checklist to create astroNN folder and save parameters
        self.pre_training_checklist_child()

        self.input_normalizer = Normalizer(mode=self.input_norm_mode)
        self.labels_normalizer = Normalizer(mode=self.labels_norm_mode)

        norm_data, self.input_mean_norm, self.input_std_norm = self.input_normalizer.normalize(input_data)
        norm_labels, self.labels_mean_norm, self.labels_std_norm = self.labels_normalizer.normalize(labels)

        self.input_shape = (norm_data.shape[1], 1,)
        self.labels_shape = norm_labels.shape[1]

        self.compile()
        self.plot_model()

        csv_logger = CSVLogger(self.fullfilepath + 'log.csv', append=True, separator=',')

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, epsilon=self.reduce_lr_epsilon,
                                      patience=self.reduce_lr_patience, min_lr=self.reduce_lr_min, mode='min',
                                      verbose=2)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=self.early_stopping_min_delta,
                                       patience=self.early_stopping_patience, verbose=2, mode='min')

        self.plot_model()

        training_generator = DataGenerator(self.batch_size).generate(norm_data, norm_labels)
        validation_generator = DataGenerator(self.batch_size).generate(norm_data, norm_labels)

        self.keras_model.fit_generator(generator=training_generator,
                                       steps_per_epoch=norm_data.shape[0] // self.batch_size,
                                       validation_data=validation_generator,
                                       validation_steps=norm_data.shape[0] // self.batch_size,
                                       epochs=self.max_epochs, max_queue_size=20, verbose=2, workers=os.cpu_count(),
                                       callbacks=[early_stopping, reduce_lr, csv_logger])

        # Call the post training checklist to save parameters
        self.post_training_checklist_child()

        return None

    def test(self, input_data):
        # Prevent shallow copy issue
        input_array = np.array(input_data)
        input_array -= self.input_mean_norm
        input_array /= self.input_std_norm
        input_array = np.atleast_3d(input_array)

        predictions = self.keras_model.predict(input_array)
        predictions *= self.labels_std_norm
        predictions += self.labels_mean_norm

        return predictions
