from abc import ABC

import numpy as np
from astroNN.datasets import H5Loader
from astroNN.models.NeuralNetMaster import NeuralNetMaster
from astroNN.nn.losses import categorical_cross_entropy, binary_cross_entropy
from astroNN.nn.losses import mean_squared_error, mean_absolute_error
from astroNN.nn.utilities import Normalizer
from astroNN.nn.utilities import categorical_accuracy, binary_accuracy
from astroNN.nn.utilities.generator import threadsafe_generator, GeneratorMaster
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


class CNNDataGenerator(GeneratorMaster):
    """
    NAME:
        CNNDataGenerator
    PURPOSE:
        To generate data for Keras
    INPUT:
    OUTPUT:
    HISTORY:
        2017-Dec-02 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, batch_size, shuffle=True):
        super(CNNDataGenerator, self).__init__(batch_size, shuffle)

    def _data_generation(self, input, labels, list_IDs_temp):
        X = self.input_d_checking(input, list_IDs_temp)
        y = labels[list_IDs_temp]

        return X, y

    @threadsafe_generator
    def generate(self, inputs, labels):
        'Generates batches of samples'
        # Infinite loop
        list_IDs = range(inputs.shape[0])
        while 1:
            # Generate order of exploration of dataset
            indexes = self._get_exploration_order(list_IDs)

            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = indexes[i * self.batch_size:(i + 1) * self.batch_size]

                # Generate data
                X, y = self._data_generation(inputs, labels, list_IDs_temp)

                yield X, y


class Pred_DataGenerator(GeneratorMaster):
    """
    NAME:
        Pred_DataGenerator
    PURPOSE:
        To generate data for Keras model prediction
    INPUT:
    OUTPUT:
    HISTORY:
        2017-Dec-02 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, batch_size, shuffle=False):
        super(Pred_DataGenerator, self).__init__(batch_size, shuffle)

    def _data_generation(self, input, list_IDs_temp):
        # Generate data
        X = self.input_d_checking(input, list_IDs_temp)

        return X

    @threadsafe_generator
    def generate(self, input):
        'Generates batches of samples'
        # Infinite loop
        list_IDs = range(input.shape[0])
        while 1:
            # Generate order of exploration of dataset
            indexes = self._get_exploration_order(list_IDs)

            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = indexes[i * self.batch_size:(i + 1) * self.batch_size]

                # Generate data
                X = self._data_generation(input, list_IDs_temp)

                yield X


class CNNBase(NeuralNetMaster, ABC):
    """Top-level class for a convolutional neural network"""

    def __init__(self):
        """
        NAME:
            __init__
        PURPOSE:
            To define astroNN convolutional neural network
        HISTORY:
            2018-Jan-06 - Written - Henry Leung (University of Toronto)
        """
        super(CNNBase, self).__init__()
        self.name = 'Convolutional Neural Network'
        self._model_type = 'CNN'
        self._model_identifier = None
        self.initializer = None
        self.activation = None
        self._last_layer_activation = None
        self.num_filters = None
        self.filter_length = None
        self.pool_length = None
        self.num_hidden = None
        self.reduce_lr_epsilon = None
        self.reduce_lr_min = None
        self.reduce_lr_patience = None
        self.l2 = None
        self.dropout_rate = 0.0
        self.val_size = 0.1

        self.input_norm_mode = 1
        self.labels_norm_mode = 2
        self.input_mean_norm = None
        self.input_std_norm = None
        self.labels_mean_norm = None
        self.labels_std_norm = None

        self.training_generator = None
        self.validation_generator = None

    def test(self, input_data):
        # Prevent shallow copy issue
        input_array = np.array(input_data)
        input_array -= self.input_mean_norm
        input_array /= self.input_std_norm

        total_test_num = input_data.shape[0]  # Number of testing data

        # for number of training data smaller than batch_size
        if input_data.shape[0] < self.batch_size:
            self.batch_size = input_data.shape[0]

        # Due to the nature of how generator works, no overlapped prediction
        data_gen_shape = (total_test_num // self.batch_size) * self.batch_size
        remainder_shape = total_test_num - data_gen_shape  # Remainder from generator

        predictions = np.zeros((total_test_num, self.labels_shape))

        # Data Generator for prediction
        prediction_generator = Pred_DataGenerator(self.batch_size).generate(input_array[:data_gen_shape])
        predictions[:data_gen_shape] = np.asarray(self.keras_model.predict_generator(
            prediction_generator, steps=input_array.shape[0] // self.batch_size))

        if remainder_shape != 0:
            remainder_data = np.atleast_3d(input_array[data_gen_shape:])
            result = self.keras_model.predict(remainder_data)
            predictions[data_gen_shape:] = result.reshape((remainder_shape, self.labels_shape))

        predictions *= self.labels_std_norm
        predictions += self.labels_mean_norm

        return predictions

    def compile(self):
        if self.optimizer is None or self.optimizer == 'adam':
            self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.optimizer_epsilon,
                                  decay=0.0)

        if self.task == 'regression':
            self._last_layer_activation = 'linear'
            loss_func = mean_squared_error
            self.metrics = [mean_absolute_error]
        elif self.task == 'classification':
            self._last_layer_activation = 'softmax'
            loss_func = categorical_cross_entropy
            self.metrics = [categorical_accuracy]
            # Don't normalize output labels for classification
            self.labels_norm_mode = 0
        elif self.task == 'multi_classification':
            self._last_layer_activation = 'sigmoid'
            loss_func = binary_cross_entropy
            self.metrics = [binary_accuracy]
            # Don't normalize output labels for classification
            self.labels_norm_mode = 0
        else:
            raise RuntimeError('Only "regression", "classification" and "multi_classification" are supported')

        self.keras_model = self.model()

        self.keras_model.compile(loss=loss_func, optimizer=self.optimizer, metrics=self.metrics)

        return None

    def pre_training_checklist_child(self, input_data, labels):
        self.pre_training_checklist_master(input_data, labels)

        if isinstance(input_data, H5Loader):
            self.targetname = input_data.target
            input_data, labels = input_data.load()

        self.input_normalizer = Normalizer(mode=self.input_norm_mode)
        self.labels_normalizer = Normalizer(mode=self.labels_norm_mode)

        norm_data, self.input_mean_norm, self.input_std_norm = self.input_normalizer.normalize(input_data)
        norm_labels, self.labels_mean_norm, self.labels_std_norm = self.labels_normalizer.normalize(labels)

        self.compile()
        self.plot_model()

        train_idx, test_idx = train_test_split(np.arange(self.num_train), test_size=self.val_size)

        self.training_generator = CNNDataGenerator(self.batch_size).generate(norm_data[train_idx],
                                                                             norm_labels[train_idx])
        self.validation_generator = CNNDataGenerator(self.batch_size).generate(norm_data[test_idx],
                                                                               norm_labels[test_idx])

        return input_data, labels

    def post_training_checklist_child(self):
        astronn_model = 'model_weights.h5'
        self.keras_model.save_weights(self.fullfilepath + astronn_model)
        print(astronn_model + ' saved to {}'.format(self.fullfilepath + astronn_model))

        np.savez(self.fullfilepath + '/astroNN_model_parameter.npz', id=self._model_identifier, pool_length=self.pool_length,
                 filterlen=self.filter_length, filternum=self.num_filters, hidden=self.num_hidden,
                 input=self.input_shape, labels=self.labels_shape, task=self.task, input_mean=self.input_mean_norm,
                 labels_mean=self.labels_mean_norm, input_std=self.input_std_norm, labels_std=self.labels_std_norm,
                 valsize=self.val_size, targetname=self.targetname, dropout_rate=self.dropout_rate, l2=self.l2,
                 input_norm_mode=self.input_norm_mode, labels_norm_mode=self.labels_norm_mode)
