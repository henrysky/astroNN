# ---------------------------------------------------------#
#   astroNN.models.starnet: Contain starnet Model
# ---------------------------------------------------------#
import os

import numpy as np
from keras.backend import clear_session
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.layers import MaxPooling1D, Conv1D, Dense, Flatten
from keras.models import Model, Input

from astroNN.models.models_shared import ModelStandard
from astroNN.models.models_tools import threadsafe_generator


class StarNet(ModelStandard):
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
        super(StarNet, self).__init__()

        self.name = 'StarNet (arXiv:1709.09182)'
        self._model_type = 'StarNet'
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
                             filters=self.num_filters[0], kernel_size=self.filter_length)(cnn_layer_1)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(cnn_layer_2)
        flattener = Flatten()(maxpool_1)
        layer_3 = Dense(units=self.num_hidden[1], kernel_initializer=self.initializer, activation=self.activation)(
            flattener)
        layer_4 = Dense(units=self.num_hidden[1], kernel_initializer=self.initializer, activation=self.activation)(
            layer_3)
        model = Model(inputs=input_tensor, outputs=layer_4)

        return model

    def compile(self):
        model = self.model()
        model.compile(loss=self.mean_squared_error, optimizer=self.optimizer)
        return model

    def train(self, x, y):
        if self.task == 'classification':
            raise RuntimeError('astroNN StarNet does not support classification task')

        x, y = super().train(x, y)

        csv_logger = CSVLogger(self.fullfilepath + 'log.csv', append=True, separator=',')

        mean_labels = np.mean(y, axis=0)
        std_labels = np.std(y, axis=0)
        mu_std = np.vstack((mean_labels, std_labels))

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=self.reduce_lr_epsilon,
                                      patience=self.reduce_lr_patience, min_lr=self.reduce_lr_min, mode='min',
                                      verbose=2)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=self.early_stopping_min_delta,
                                       patience=self.early_stopping_patience, verbose=2, mode='min')

        model = self.compile()

        self.plot_model(model)

        training_generator = DataGenerator(x.shape[1], self.batch_size).generate(x, y)

        model.fit_generator(generator=training_generator, steps_per_epoch=x.shape[0] // self.batch_size,
                            epochs=self.max_epochs, max_queue_size=20, verbose=2, workers=os.cpu_count(),
                            callbacks=[early_stopping, reduce_lr, csv_logger])

        astronn_model = 'model.h5'
        model.save(self.fullfilepath + astronn_model)
        print(astronn_model + ' saved to {}'.format(self.fullfilepath + astronn_model))
        np.save(self.fullfilepath + 'meanstd.npy', mu_std)
        np.save(self.fullfilepath + 'targetname.npy', self.target)

        return None

    def test(self, x):
        x, model = super().test(x)
        return model.predict(x)


class DataGenerator(object):
    """
    NAME:
        DataGenerator
    PURPOSE:
        To generate data for Keras
    INPUT:
    OUTPUT:
    HISTORY:
        2017-Dec-02 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, dim, batch_size, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __get_exploration_order(self, list_IDs):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle is True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, spectra, labels, list_IDs_temp):
        'Generates data of batch_size samples'
        # X : (n_samples, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim, 1))
        y = np.empty((self.batch_size, labels.shape[1]))

        # Generate data
        X[:, :, 0] = spectra[list_IDs_temp]
        y[:] = labels[list_IDs_temp]

        return X, y

    @threadsafe_generator
    def generate(self, input, labels):
        'Generates batches of samples'
        # Infinite loop
        list_IDs = range(input.shape[0])
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)

            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = indexes[i * self.batch_size:(i + 1) * self.batch_size]

                # Generate data
                X, y = self.__data_generation(input, labels, list_IDs_temp)

                yield (X, y)
