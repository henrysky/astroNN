import time
from abc import ABC
import numpy as np
from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
import keras.backend as K

from astroNN.datasets import H5Loader
from astroNN.models.NeuralNetMaster import NeuralNetMaster
from astroNN.models.losses.classification import categorical_cross_entropy, bayes_crossentropy_wrapper
from astroNN.models.losses.regression import mean_squared_error, mean_absolute_error
from astroNN.models.utilities.generator import threadsafe_generator, GeneratorMaster
from astroNN.models.utilities.normalizer import Normalizer
from astroNN.models.utilities.metrics import categorical_accuracy


class Bayesian_DataGenerator(GeneratorMaster):
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
        super(Bayesian_DataGenerator, self).__init__(batch_size, shuffle)

    def _data_generation(self, input, labels, labels_err, list_IDs_temp):
        X = self.input_d_checking(input, list_IDs_temp)
        y = np.empty((self.batch_size, labels.shape[1]))
        y_err = np.empty((self.batch_size, labels.shape[1]))
        y[:] = labels[list_IDs_temp]
        y_err[:] = labels_err[list_IDs_temp]

        return X, y, y_err

    @threadsafe_generator
    def generate(self, input, labels, labels_err):
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
                X, y, y_err = self._data_generation(input, labels, labels_err, list_IDs_temp)

                yield {'input': X, 'labels_err': y_err}, {'output': y, 'variance_output': y}


class Bayesian_Pred_DataGenerator(GeneratorMaster):
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
        super(Bayesian_Pred_DataGenerator, self).__init__(batch_size, shuffle)

    def _data_generation(self, input, input_err, list_IDs_temp):
        'Generates data of batch_size samples'
        # X : (n_samples, v_size, n_channels)
        # Initialization
        X = self.input_d_checking(input, list_IDs_temp)
        X_err = self.input_d_checking(input_err, list_IDs_temp)

        # Generate data
        X += X_err

        return X

    @threadsafe_generator
    def generate(self, inputs, input_err):
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
                X = self._data_generation(inputs, input_err, list_IDs_temp)

                yield X


class BayesianCNNBase(NeuralNetMaster, ABC):
    """Top-level class for a Bayesian convolutional neural network"""

    def __init__(self):
        """
        NAME:
            __init__
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
        self.length_scale = 3  # prior length scale
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

        self.keras_model_predict = None

        K.set_learning_phase(1)

    def test(self, input_data, inputs_err):
        """
        NAME:
            test
        PURPOSE:
            test model
        HISTORY:
            2018-Jan-06 - Written - Henry Leung (University of Toronto)
        """
        # Prevent shallow copy issue
        input_array = np.array(input_data)
        input_array -= self.input_mean_norm
        input_array /= self.input_std_norm

        K.set_learning_phase(1)

        total_test_num = input_data.shape[0]  # Number of testing data

        predictions = np.zeros((self.mc_num, total_test_num, self.labels_shape))
        predictions_var = np.zeros((self.mc_num, total_test_num, self.labels_shape))

        # Due to the nature of how generator works, no overlapped prediction
        data_gen_shape = (total_test_num // self.batch_size) * self.batch_size
        remainder_shape = total_test_num - data_gen_shape  # Remainder from generator

        start_time = time.time()

        for counter, i in enumerate(range(self.mc_num)):
            if counter % 5 == 0:
                print('Completed {} of {} Monte Carlo, {:.03f} seconds elapsed'.format(counter, self.mc_num,
                                                                                       time.time() - start_time))

            # Data Generator for prediction
            prediction_generator = Bayesian_Pred_DataGenerator(self.batch_size).generate(input_array[:data_gen_shape],
                                                                                         inputs_err[:data_gen_shape])

            result = np.asarray(self.keras_model_predict.predict_generator(
                prediction_generator, steps=data_gen_shape // self.batch_size))

            predictions[i, :data_gen_shape] = result[0].reshape((data_gen_shape, self.labels_shape))
            predictions_var[i, :data_gen_shape] = result[1].reshape((data_gen_shape, self.labels_shape))

            if remainder_shape != 0:
                remainder_data = np.atleast_3d(input_array[data_gen_shape:] +
                                               np.random.normal(0, inputs_err[data_gen_shape:]))
                result = self.keras_model_predict.predict(remainder_data)
                predictions[i, data_gen_shape:] = result[0].reshape((remainder_shape, self.labels_shape))
                predictions_var[i, data_gen_shape:] = result[1].reshape((remainder_shape, self.labels_shape))

        print('Completed Dropout Variational Inference, {:.03} seconds in total'.format(time.time() - start_time))

        predictions *= self.labels_std_norm
        predictions += self.labels_mean_norm

        pred = np.mean(predictions, axis=0)
        var_mc_dropout = np.var(predictions, axis=0)

        var = np.mean(np.exp(predictions_var) * (self.labels_std_norm ** 2), axis=0)
        pred_var = var + var_mc_dropout + self.inv_model_precision  # epistemic plus aleatoric uncertainty plus tau
        pred_std = np.sqrt(pred_var)  # Convert back to std error
        print(self.inv_model_precision)

        print(pred_var)
        print(var)

        print('Finished testing!')

        return pred, pred_std

    def compile(self):
        if self.task == 'regression':
            self._last_layer_activation = 'linear'
        elif self.task == 'classification':
            self._last_layer_activation = 'softmax'

        self.keras_model, self.keras_model_predict, output_loss, variance_loss = self.model()

        if self.optimizer is None or self.optimizer == 'adam':
            self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.optimizer_epsilon,
                                  decay=0.0)

        if self.task == 'regression':
            self.metrics = mean_absolute_error
            self.keras_model.compile(loss={'output': output_loss,
                                           'variance_output': variance_loss},
                                     optimizer=self.optimizer,
                                     loss_weights={'output': .5, 'variance_output': .5},
                                     metrics={'output': self.metrics})
        elif self.task == 'classification':
            print('Currently Not Working Properly')
            self.metrics = categorical_accuracy
            self.keras_model.compile(loss={'output': categorical_cross_entropy,
                                           'variance_output': bayes_crossentropy_wrapper},
                                     optimizer=self.optimizer,
                                     loss_weights={'output': 1., 'variance_output': .1},
                                     metrics={'output': self.metrics})
        else:
            raise RuntimeError('Only "regression" and "classification" are supported')

        return None

    def pre_training_checklist_child(self, input_data, labels, labels_err):
        self.pre_training_checklist_master(input_data, labels)

        if isinstance(input_data, H5Loader):
            self.targetname = input_data.target
            input_data, labels = input_data.load()

        if labels_err is None:
            labels_err = np.zeros(labels.shape)

        self.input_normalizer = Normalizer(mode=self.input_norm_mode)
        self.labels_normalizer = Normalizer(mode=self.labels_norm_mode)

        norm_data, self.input_mean_norm, self.input_std_norm = self.input_normalizer.normalize(input_data)
        norm_labels, self.labels_mean_norm, self.labels_std_norm = self.labels_normalizer.normalize(labels)
        norm_labels_err = labels_err / self.labels_std_norm

        self.compile()
        self.plot_model()

        train_idx, test_idx = train_test_split(np.arange(self.num_train), test_size=self.val_size)

        self.inv_model_precision = (2 * self.num_train * self.l2) / (self.length_scale ** 2 * (1 - self.dropout_rate))

        self.training_generator = Bayesian_DataGenerator(self.batch_size).generate(norm_data[train_idx],
                                                                                   norm_labels[train_idx],
                                                                                   norm_labels_err[train_idx])
        self.validation_generator = Bayesian_DataGenerator(self.batch_size).generate(norm_data[test_idx],
                                                                                     norm_labels[test_idx],
                                                                                     norm_labels_err[test_idx])

        return input_data, labels, labels_err

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
