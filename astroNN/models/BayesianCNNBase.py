import time
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import numpy as np

import keras.backend as K
from keras.optimizers import Adam

from astroNN.datasets import H5Loader
from astroNN.models.NeuralNetMaster import NeuralNetMaster
from astroNN.models.loss.classification import categorical_cross_entropy, bayes_crossentropy_wrapper
from astroNN.models.loss.regression import mean_squared_error, mean_absolute_error
from astroNN.models.utilities.generator import threadsafe_generator
from astroNN.models.utilities.normalizer import Normalizer

class Bayesian_DataGenerator(object):
    """
    NAME:
        Bayesian_DataGenerator
    PURPOSE:
        To generate data for Keras
    INPUT:
    OUTPUT:
    HISTORY:
        2017-Dec-02 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, batch_size, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __get_exploration_order(self, list_IDs):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle is True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, input, labels, list_IDs_temp):
        'Generates data of batch_size samples'
        # X : (n_samples, v_size, n_channels)
        # Initialization
        if input.ndim == 2:
            X = np.empty((self.batch_size, input.shape[1], 1))
            y = np.empty((self.batch_size, labels.shape[1]))
            # Generate data
            X[:, :, 0] = input[list_IDs_temp]
            y[:] = labels[list_IDs_temp]

        elif input.ndim == 3:
            X = np.empty((self.batch_size, input.shape[1], input.shape[2], 1))
            y = np.empty((self.batch_size, labels.shape[1]))
            # Generate data
            X[:, :, :, 0] = input[list_IDs_temp]
            y[:] = labels[list_IDs_temp]

        elif input.ndim == 4:
            X = np.empty((self.batch_size, input.shape[1], input.shape[2], input.shape[3]))
            y = np.empty((self.batch_size, labels.shape[1]))
            # Generate data
            X[:, :, :, :] = input[list_IDs_temp]
            y[:] = labels[list_IDs_temp]
        else:
            raise TypeError

        return X, y

    def sparsify(self, y):
        'Returns labels in binary NumPy array'
        # n_classes =  # Enter number of classes
        # return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
        #                  for i in range(y.shape[0])])
        pass

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

                yield (X, {'output': y, 'variance_output': y})


class BayesianCNNBase(NeuralNetMaster, ABC):
    """Top-level class for a Bayesian convolutional neural network"""

    def __init__(self):
        """
        NAME:
            __init__on: {} \n".format(self._implementation_version))
            h.write("python versi
        PURPOSE:
            To define astroNN Bayesian convolutional neural network
        HISTORY:
            2018-Jan-06 - Written - Henry Leung (University of Toronto)
        """
        super(BayesianCNNBase, self).__init__()
        self.name = 'Bayesian Convolutional Neural Network'
        self._model_type = 'BCNN'
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
        self.inv_model_precision = None  # inverse model precision
        self.dropout_rate = 0.2
        self.length_scale = 0.1  # prior length scale
        self.mc_num = 25
        self.val_size = 0.1

        self.input_norm_mode = 1
        self.labels_norm_mode = 2
        self.input_mean_norm = None
        self.input_std_norm = None
        self.labels_mean_norm = None
        self.labels_std_norm = None

        self.training_generator = None
        self.validation_generator = None

        K.set_learning_phase(1)

    @abstractmethod
    def model(self):
        raise NotImplementedError

    @abstractmethod
    def train(self, input_data, labels, inputs_err, labels_err):
        raise NotImplementedError

    def test(self, input_data, inputs_err):
        # Prevent shallow copy issue
        input_array = np.array(input_data)
        input_array -= self.input_mean_norm
        input_array /= self.input_std_norm
        input_array = np.atleast_3d(input_array)

        K.set_learning_phase(1)

        predictions = np.zeros((self.mc_num, input_array.shape[0], self.labels_shape))
        predictions_var = np.zeros((self.mc_num, input_array.shape[0], self.labels_shape))

        start_time = time.time()

        for counter, i in enumerate(range(self.mc_num)):
            if counter % 5 == 0:
                print('Completed {} of {} Monte Carlo, {:.03f} seconds elapsed'.format(counter + 1, self.mc_num,
                                                                                       time.time() - start_time))
            input_array_with_error = input_array + np.atleast_3d(np.random.normal(0, inputs_err))
            result = np.asarray(self.keras_model.predict(input_array_with_error))
            predictions[i] = result[0].reshape((input_array.shape[0], self.labels_shape))
            predictions_var[i] = result[1].reshape((input_array.shape[0], self.labels_shape))

        predictions *= self.labels_std_norm
        predictions += self.labels_mean_norm

        pred = np.mean(predictions, axis=0)
        var_mc_dropout = np.var(predictions, axis=0)

        var = np.mean(np.exp(predictions_var) * self.labels_std_norm, axis=0)
        pred_var = var + var_mc_dropout + self.inv_model_precision  # epistemic plus aleatoric uncertainty plus tau
        pred_var = np.sqrt(pred_var)  # Convert back to std error
        print(self.inv_model_precision)
        print(pred_var)

        print(var)

        print('Finished testing!')

        # self.aspcap_residue_plot(pred, y, pred_var)
        return pred, pred_var

    def compile(self):
        self.keras_model, output_loss, variance_loss = self.model()

        if self.optimizer is None or self.optimizer == 'adam':
            self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.optimizer_epsilon,
                                  decay=0.0)

        if self.task == 'regression':
            self._last_layer_activation = 'linear'
            self.metrics = mean_absolute_error
            self.keras_model.compile(loss={'output': mean_squared_error,
                                           'variance_output': variance_loss},
                                     optimizer=self.optimizer,
                                     loss_weights={'output': 1., 'variance_output': .1},
                                     metrics={'output': self.metrics})
        elif self.task == 'classification':
            print('Currently Not Working Properly')
            self._last_layer_activation = 'softmax'
            self.metrics = 'accuracy'
            self.keras_model.compile(loss={'output': categorical_cross_entropy,
                                           'variance_output': bayes_crossentropy_wrapper(100, 10)},
                                     optimizer=self.optimizer,
                                     loss_weights={'output': 1., 'variance_output': .1},
                                     metrics={'output': self.metrics})
        else:
            raise RuntimeError('Only "regression" and "classification" are supported')

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

        self.inv_model_precision = (2 * self.num_train * self.l2) / (self.length_scale ** 2 * (1 - self.dropout_rate))

        self.training_generator = Bayesian_DataGenerator(self.batch_size).generate(norm_data[train_idx], norm_labels[train_idx])
        self.validation_generator = Bayesian_DataGenerator(self.batch_size).generate(norm_data[test_idx], norm_labels[test_idx])

        return input_data, labels

    def post_training_checklist_child(self):
        astronn_model = 'model_weights.h5'
        self.keras_model.save_weights(self.fullfilepath + astronn_model)
        print(astronn_model + ' saved to {}'.format(self.fullfilepath + astronn_model))

        np.savez(self.fullfilepath + '/astroNN_model_parameter.npz', id=self._model_identifier,
                 filterlen=self.filter_length, filternum=self.num_filters, hidden=self.num_hidden,
                 input=self.input_shape, labels=self.labels_shape, task=self.task, inv_tau=self.inv_model_precision,
                 input_mean=self.input_mean_norm, labels_mean=self.labels_mean_norm, input_std=self.input_std_norm,
                 valsize=self.val_size, labels_std=self.labels_std_norm, targetname=self.targetname)

        K.clear_session()
