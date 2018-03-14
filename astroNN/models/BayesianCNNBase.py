import os
import time
from abc import ABC
import json
import numpy as np
from sklearn.model_selection import train_test_split

from astroNN.config import MULTIPROCESS_FLAG
from astroNN.config import keras_import_manager
from astroNN.datasets import H5Loader
from astroNN.models.NeuralNetMaster import NeuralNetMaster
from astroNN.nn.callbacks import VirutalCSVLogger
from astroNN.nn.losses import mean_absolute_error
from astroNN.nn.metrics import categorical_accuracy
from astroNN.nn.utilities import Normalizer
from astroNN.nn.utilities.generator import threadsafe_generator, GeneratorMaster

keras = keras_import_manager()
regularizers = keras.regularizers
ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
Adam = keras.optimizers.Adam


class BayesianCNNDataGenerator(GeneratorMaster):
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
        super(BayesianCNNDataGenerator, self).__init__(batch_size, shuffle)

    def _data_generation(self, inputs, labels, input_err, labels_err, idx_list_temp):
        x = self.input_d_checking(inputs, idx_list_temp)
        x_err = self.input_d_checking(input_err, idx_list_temp)
        y = labels[idx_list_temp]
        y_err = labels_err[idx_list_temp]

        return x, y, x_err, y_err

    @threadsafe_generator
    def generate(self, inputs, labels, input_err, labels_err):
        # Infinite loop
        idx_list = range(inputs.shape[0])
        while 1:
            # Generate order of exploration of dataset
            indexes = self._get_exploration_order(idx_list)

            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                idx_list_temp = indexes[i * self.batch_size:(i + 1) * self.batch_size]

                # Generate data
                x, y, x_err, y_err = self._data_generation(inputs, labels, input_err, labels_err, idx_list_temp)

                yield {'input': x, 'labels_err': y_err, 'input_err': x_err}, {'output': y, 'variance_output': y}


class BayesianCNNPredDataGenerator(GeneratorMaster):
    """
    NAME:
        BayesianCNNPredDataGenerator
    PURPOSE:
        To generate data for Keras model prediction
    INPUT:
    OUTPUT:
    HISTORY:
        2017-Dec-02 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, batch_size, shuffle=False):
        super(BayesianCNNPredDataGenerator, self).__init__(batch_size, shuffle)

    def _data_generation(self, inputs, input_err, idx_list_temp):
        # X : (n_samples, v_size, n_channels)
        # Initialization
        x = self.input_d_checking(inputs, idx_list_temp)
        x_err = self.input_d_checking(input_err, idx_list_temp)

        # No need to generate new spectra here anymore, migrated to be done with tensorflow (possibly GPU)

        return x, x_err

    @threadsafe_generator
    def generate(self, inputs, input_err):
        # Infinite loop
        idx_list = range(inputs.shape[0])
        while 1:
            # Generate order of exploration of dataset
            indexes = self._get_exploration_order(idx_list)

            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                idx_list_temp = indexes[i * self.batch_size:(i + 1) * self.batch_size]

                # Generate data
                x, x_err = self._data_generation(inputs, input_err, idx_list_temp)

                yield {'input': x, 'input_err': x_err}


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
        self.filter_len = None
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
        self.diable_dropout = False

        self.input_norm_mode = 1
        self.labels_norm_mode = 2

        self.keras_model_predict = None

    def test(self, input_data, inputs_err=None):
        """
        NAME:
            tests
        PURPOSE:
            tests model
        HISTORY:
            2018-Jan-06 - Written - Henry Leung (University of Toronto)
        """
        self.pre_testing_checklist_master()

        # Prevent shallow copy issue
        input_array = np.array(input_data)
        input_array -= self.input_mean
        input_array /= self.input_std

        # if no error array then just zeros
        if inputs_err is None:
            inputs_err = np.zeros_like(input_data)
        else:
            inputs_err /= self.input_std

        total_test_num = input_data.shape[0]  # Number of testing data

        # for number of training data smaller than batch_size
        if input_data.shape[0] < self.batch_size:
            self.batch_size = input_data.shape[0]

        predictions = np.zeros((self.mc_num, total_test_num, self.labels_shape))
        predictions_var = np.zeros((self.mc_num, total_test_num, self.labels_shape))

        # Due to the nature of how generator works, no overlapped prediction
        data_gen_shape = (total_test_num // self.batch_size) * self.batch_size
        remainder_shape = total_test_num - data_gen_shape  # Remainder from generator

        start_time = time.time()
        print("Starting Dropout Variational Inference")
        for counter, i in enumerate(range(self.mc_num)):
            if counter % 5 == 0:
                print('Completed {} of {} Monte Carlo Dropout, {:.03f}s elapsed'.format(counter, self.mc_num,
                                                                                        time.time() - start_time))

            # Data Generator for prediction
            prediction_generator = BayesianCNNPredDataGenerator(self.batch_size).generate(input_array[:data_gen_shape],
                                                                                          inputs_err[:data_gen_shape])

            result = np.asarray(self.keras_model_predict.predict_generator(
                prediction_generator, steps=data_gen_shape // self.batch_size))

            predictions[i, :data_gen_shape] = result[0].reshape((data_gen_shape, self.labels_shape))
            predictions_var[i, :data_gen_shape] = result[1].reshape((data_gen_shape, self.labels_shape))

            if remainder_shape != 0:
                remainder_data = input_array[data_gen_shape:]
                remainder_data_err = inputs_err[data_gen_shape:]
                # assume its caused by mono images, so need to expand dim by 1
                if len(input_array[0].shape) != len(self.input_shape):
                    remainder_data = np.expand_dims(remainder_data, axis=-1)
                    remainder_data_err = np.expand_dims(remainder_data_err, axis=-1)
                result = self.keras_model_predict.predict({'input': remainder_data, 'input_err': remainder_data_err})
                predictions[i, data_gen_shape:] = result[0].reshape((remainder_shape, self.labels_shape))
                predictions_var[i, data_gen_shape:] = result[1].reshape((remainder_shape, self.labels_shape))

        print('Completed Dropout Variational Inference, {:.03f} seconds in total'.format(time.time() - start_time))

        predictions *= self.labels_std
        predictions += self.labels_mean

        pred = np.mean(predictions, axis=0)
        var_mc_dropout = np.var(predictions, axis=0)

        if self.task == 'regression':
            # Predictive variance
            var = np.mean(np.exp(predictions_var) * (np.array(self.labels_std) ** 2), axis=0)
            pred_var = var + var_mc_dropout  # epistemic plus aleatoric uncertainty
            pred_std = np.sqrt(pred_var)  # Convert back to std error
        elif self.task == 'classification':
            pred_var = np.var(predictions, axis=0)
            pred_std = pred_var
            var = 1
        else:
            raise AttributeError('Unknown Task')

        return pred, {'total': pred_std, 'model': np.sqrt(var), 'predictive': np.sqrt(var_mc_dropout)}

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None):
        if self.task == 'regression':
            self._last_layer_activation = 'linear'
        elif self.task == 'classification':
            self._last_layer_activation = 'softmax'

        self.keras_model, self.keras_model_predict, output_loss, variance_loss = self.model()

        if optimizer is not None:
            self.optimizer = optimizer
        elif self.optimizer is None or self.optimizer == 'adam':
            self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.optimizer_epsilon,
                                  decay=0.0)
        if self.task == 'regression':
            self.metrics = [mean_absolute_error]
            self.keras_model.compile(loss={'output': output_loss,
                                           'variance_output': variance_loss},
                                     optimizer=self.optimizer,
                                     loss_weights={'output': .5, 'variance_output': .5},
                                     metrics={'output': self.metrics})
        elif self.task == 'classification':
            print('Currently Not Working Properly')
            self.metrics = [categorical_accuracy]
            self.keras_model.compile(loss={'variance_output': output_loss},
                                     optimizer=self.optimizer,
                                     loss_weights={'variance_output': 1.},
                                     metrics={'output': self.metrics})
        else:
            raise RuntimeError('Only "regression" and "classification" are supported')

        return None

    def pre_training_checklist_child(self, input_data, labels, input_err, labels_err):
        self.pre_training_checklist_master(input_data, labels)

        if isinstance(input_data, H5Loader):
            self.targetname = input_data.target
            input_data, labels = input_data.load()

        self.input_normalizer = Normalizer(mode=self.input_norm_mode)
        self.labels_normalizer = Normalizer(mode=self.labels_norm_mode)

        norm_data = self.input_normalizer.normalize(input_data)
        self.input_mean, self.input_std = self.input_normalizer.mean_labels, self.input_normalizer.std_labels
        norm_labels = self.labels_normalizer.normalize(labels)
        self.labels_mean, self.labels_std = self.labels_normalizer.mean_labels, self.labels_normalizer.std_labels

        # No need to care about Magic number as loss function looks for magic num in y_true only
        norm_input_err = input_err / self.input_std
        norm_labels_err = labels_err / self.labels_std

        self.compile()

        train_idx, test_idx = train_test_split(np.arange(self.num_train), test_size=self.val_size)

        self.inv_model_precision = (2 * self.num_train * self.l2) / (self.length_scale ** 2 * (1 - self.dropout_rate))

        self.training_generator = BayesianCNNDataGenerator(self.batch_size).generate(norm_data[train_idx],
                                                                                     norm_labels[train_idx],
                                                                                     norm_input_err[train_idx],
                                                                                     norm_labels_err[train_idx])
        self.validation_generator = BayesianCNNDataGenerator(self.batch_size).generate(norm_data[test_idx],
                                                                                       norm_labels[test_idx],
                                                                                       norm_input_err[test_idx],
                                                                                       norm_labels_err[test_idx])

        return input_data, labels, labels_err

    def post_training_checklist_child(self):
        astronn_model = 'model_weights.h5'
        self.keras_model.save(self.fullfilepath + astronn_model)
        print(astronn_model + ' saved to {}'.format(self.fullfilepath + astronn_model))

        self.hyper_txt.write("Dropout Rate: {} \n".format(self.dropout_rate))
        self.hyper_txt.flush()
        self.hyper_txt.close()

        data = {'id': self.__class__.__name__, 'pool_length': self.pool_length, 'filterlen': self.filter_len,
                'filternum': self.num_filters, 'hidden': self.num_hidden, 'input': self.input_shape,
                'labels': self.labels_shape, 'task': self.task, 'input_mean': self.input_mean.tolist(),
                'inv_tau': self.inv_model_precision, 'length_scale': self.length_scale,
                'labels_mean': self.labels_mean.tolist(), 'input_std': self.input_std.tolist(),
                'labels_std': self.labels_std.tolist(),
                'valsize': self.val_size, 'targetname': self.targetname, 'dropout_rate': self.dropout_rate,
                'l2': self.l2, 'input_norm_mode': self.input_norm_mode, 'labels_norm_mode': self.labels_norm_mode,
                'batch_size': self.batch_size}

        with open(self.fullfilepath + '/astroNN_model_parameter.json', 'w') as f:
            json.dump(data, f, indent=4, sort_keys=True)

    def train(self, input_data, labels, inputs_err=None, labels_err=None):
        if inputs_err is None:
            inputs_err = np.zeros_like(input_data)

        if labels_err is None:
            labels_err = np.zeros_like(labels)

        # Call the checklist to create astroNN folder and save parameters
        self.pre_training_checklist_child(input_data, labels, inputs_err, labels_err)

        reduce_lr = ReduceLROnPlateau(monitor='val_output_loss', factor=0.5, epsilon=self.reduce_lr_epsilon,
                                      patience=self.reduce_lr_patience, min_lr=self.reduce_lr_min, mode='min',
                                      verbose=2)

        self.virtual_cvslogger = VirutalCSVLogger()

        self.history = self.keras_model.fit_generator(generator=self.training_generator,
                                                      steps_per_epoch=self.num_train // self.batch_size,
                                                      validation_data=self.validation_generator,
                                                      validation_steps=self.val_num // self.batch_size,
                                                      epochs=self.max_epochs, verbose=self.verbose,
                                                      workers=os.cpu_count(),
                                                      callbacks=[reduce_lr, self.virtual_cvslogger],
                                                      use_multiprocessing=MULTIPROCESS_FLAG)

        if self.autosave is True:
            # Call the post training checklist to save parameters
            self.save()

        return None
