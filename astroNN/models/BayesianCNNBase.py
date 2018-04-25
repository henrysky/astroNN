import json
import os
import time
from abc import ABC

import numpy as np
from astroNN.config import MULTIPROCESS_FLAG
from astroNN.config import keras_import_manager
from astroNN.datasets import H5Loader
from astroNN.models.NeuralNetMaster import NeuralNetMaster
from astroNN.nn.callbacks import VirutalCSVLogger
from astroNN.nn.layers import FastMCInference
from astroNN.nn.losses import mean_absolute_error
from astroNN.nn.metrics import categorical_accuracy, binary_accuracy
from astroNN.nn.utilities import Normalizer
from astroNN.nn.utilities.generator import threadsafe_generator, GeneratorMaster
from astroNN.shared.nn_tools import get_available_gpus
from sklearn.model_selection import train_test_split

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
        super().__init__(batch_size, shuffle)

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
        super().__init__(batch_size, shuffle)

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
    """
    Top-level class for a Bayesian convolutional neural network

    :History: 2018-Jan-06 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self):
        super().__init__()
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
        self.mc_num = 100  # increased to 100 due to high performance VI on GPU implemented on 14 April 2018 (Henry)
        self.val_size = 0.1
        self.disable_dropout = False

        self.input_norm_mode = 1
        self.labels_norm_mode = 2

        self.keras_model_predict = None

    def test(self, input_data, inputs_err=None):
        """
        High performance version designed for fast variational inference on GPU

        :param input_data: Data to be inferred with neural network
        :type input_data: ndarray
        :param inputs_err: Error for input_data, same shape with input_data.
        :type inputs_err: Union[None, ndarray]
        :param model_plot: True to plot model too
        :type model_plot: boolean
        :return: A saved folder on disk
        :History:
            | 2018-Jan-06 - Written - Henry Leung (University of Toronto)
            | 2018-Apr-12 - Updated - Henry Leung (University of Toronto)
        """
        if get_available_gpus() is False and self.mc_num > 25:
            print(f'You are using CPU version Tensorflow, doing {self.mc_num} times Monte Carlo Inference can '
                  f'potentially be very slow!')
            print('A possible fix is to decrease the mc_num parameter of the model to do less Monte Carlo Inference')

        self.pre_testing_checklist_master()

        input_data = np.atleast_2d(input_data)

        if self.input_normalizer is not None:
            input_array = self.input_normalizer.normalize(input_data, calc=False)
        else:
            # Prevent shallow copy issue
            input_array = np.array(input_data)
            input_array -= self.input_mean
            input_array /= self.input_std

        # if no error array then just zeros
        if inputs_err is None:
            inputs_err = np.zeros_like(input_data)
        else:
            inputs_err = np.atleast_2d(inputs_err)
            inputs_err /= self.input_std

        total_test_num = input_data.shape[0]  # Number of testing data

        # for number of training data smaller than batch_size
        if total_test_num < self.batch_size:
            self.batch_size = total_test_num

        # Due to the nature of how generator works, no overlapped prediction
        data_gen_shape = (total_test_num // self.batch_size) * self.batch_size
        remainder_shape = total_test_num - data_gen_shape  # Remainder from generator

        start_time = time.time()
        print("Starting Dropout Variational Inference")

        # Data Generator for prediction
        prediction_generator = BayesianCNNPredDataGenerator(self.batch_size).generate(input_array[:data_gen_shape],
                                                                                      inputs_err[:data_gen_shape])

        new = FastMCInference(self.mc_num)(self.keras_model_predict)

        result = np.asarray(new.predict_generator(prediction_generator, steps=data_gen_shape // self.batch_size))

        if remainder_shape != 0:  # deal with remainder
            remainder_generator = BayesianCNNPredDataGenerator(remainder_shape).generate(input_array[data_gen_shape:],
                                                                                          inputs_err[data_gen_shape:])
            remainder_result = np.asarray(new.predict_generator(remainder_generator, steps=1))
            result = np.concatenate((result, remainder_result))

        if result.ndim < 3:  # in case only 1 test data point, in such case we need to add a dimension
            result = np.expand_dims(result, axis=0)

        half_first_dim = result.shape[1] // 2  # result.shape[1] is guarantee an even number, otherwise sth is wrong

        predictions = result[:, :half_first_dim, 0]  # mean prediction
        mc_dropout_uncertainty = result[:, :half_first_dim, 1] * (self.labels_std ** 2)  # model uncertainty
        predictions_var = np.exp(result[:, half_first_dim:, 0]) * (self.labels_std ** 2)  # predictive uncertainty

        print(f'Completed Dropout Variational Inference with {self.mc_num} forward passes, '
              f'{(time.time() - start_time):.{2}f}s in total')

        if self.labels_normalizer is not None:
            predictions = self.labels_normalizer.denormalize(predictions)
        else:
            predictions *= self.labels_std
            predictions += self.labels_mean

        if self.task == 'regression':
            # Predictive variance
            pred_var = predictions_var + mc_dropout_uncertainty  # epistemic plus aleatoric uncertainty
            pred_uncertainty = np.sqrt(pred_var)  # Convert back to std error

            # final correction from variance to standard derivation
            mc_dropout_uncertainty = np.sqrt(mc_dropout_uncertainty)
            predictive_uncertainty = np.sqrt(predictions_var)

        elif self.task == 'classification':
            # we want entropy for classification uncertainty
            predictions = np.argmax(predictions, axis=1)
            mc_dropout_uncertainty_temp = np.array(mc_dropout_uncertainty)
            mc_dropout_uncertainty = np.ones_like(predictions, dtype=float)
            predictive_uncertainty = np.ones_like(predictions, dtype=float)
            for i in range(predictions.shape[0]):
                mc_dropout_uncertainty[i] = mc_dropout_uncertainty_temp[i, predictions[i]]
                predictive_uncertainty[i] = np.array(predictions_var[i, predictions[i]])

            pred_uncertainty = mc_dropout_uncertainty + predictive_uncertainty

        elif self.task == 'binary_classification':
            # we want entropy for classification uncertainty
            mc_dropout_uncertainty = - np.sum(predictions * np.log(predictions), axis=0)  # need to use raw prediction for uncertainty
            predictions = np.rint(predictions)
            predictive_uncertainty = predictions_var
            pred_uncertainty = mc_dropout_uncertainty + predictions_var
        else:
            raise AttributeError('Unknown Task')

        return predictions, {'total': pred_uncertainty, 'model': mc_dropout_uncertainty, 'predictive': predictive_uncertainty}

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None):
        if optimizer is not None:
            self.optimizer = optimizer
        elif self.optimizer is None or self.optimizer == 'adam':
            self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.optimizer_epsilon,
                                  decay=0.0)

        if self.task == 'regression':
            self._last_layer_activation = 'linear'
        elif self.task == 'classification':
            self._last_layer_activation = 'softmax'
        elif self.task == 'binary_classification':
            self._last_layer_activation = 'sigmoid'
        else:
            raise RuntimeError('Only "regression", "classification" and "binary_classification" are supported')

        self.keras_model, self.keras_model_predict, output_loss, variance_loss = self.model()

        if self.task == 'regression':
            if self.metrics is None:
                self.metrics = [mean_absolute_error]
            self.keras_model.compile(loss={'output': output_loss, 'variance_output': variance_loss},
                                     optimizer=self.optimizer,
                                     loss_weights={'output': .5, 'variance_output': .5},
                                     metrics={'output': self.metrics})
        elif self.task == 'classification':
            print('Sorry but there is a known issue of the loss not handling loss correctly. I will fix it in May'
                  '-- Henry 19 April 2018')
            if self.metrics is None:
                self.metrics = [categorical_accuracy]
            self.keras_model.compile(loss={'output': output_loss, 'variance_output': variance_loss},
                                     optimizer=self.optimizer,
                                     loss_weights={'output': .5, 'variance_output': .5},
                                     metrics={'output': self.metrics})
        elif self.task == 'binary_classification':
            print('Sorry but there is a known issue of the loss not handling loss correctly. I will fix it in May'
                  '-- Henry 19 April 2018')
            if self.metrics is None:
                self.metrics = [binary_accuracy(from_logits=True)]
            self.keras_model.compile(loss={'output': output_loss, 'variance_output': variance_loss},
                                     optimizer=self.optimizer,
                                     loss_weights={'output': .5, 'variance_output': .5},
                                     metrics={'output': self.metrics})
        return None

    def test_old(self, input_data, inputs_err=None):
        """
        NAME:
            test_old
        PURPOSE:
            tests model
        HISTORY:
            2018-Jan-06 - Written - Henry Leung (University of Toronto)
        """
        self.pre_testing_checklist_master()

        if self.input_normalizer is not None:
            input_array = self.input_normalizer.normalize(input_data, calc=False)
        else:
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
        if total_test_num < self.batch_size:
            self.batch_size = total_test_num

        predictions = np.zeros((self.mc_num, total_test_num, self.labels_shape))
        predictions_var = np.zeros((self.mc_num, total_test_num, self.labels_shape))

        # Due to the nature of how generator works, no overlapped prediction
        data_gen_shape = (total_test_num // self.batch_size) * self.batch_size
        remainder_shape = total_test_num - data_gen_shape  # Remainder from generator

        start_time = time.time()
        print("Starting Dropout Variational Inference")
        for i in range(self.mc_num):
            if i % 5 == 0:
                print(f'Completed {i} of {self.mc_num} Monte Carlo Dropout, {(time.time() - start_time):.{2}f}s '
                      f'elapsed')

            # Data Generator for prediction
            prediction_generator = BayesianCNNPredDataGenerator(self.batch_size).generate(input_array[:data_gen_shape],
                                                                                          inputs_err[:data_gen_shape])

            result = np.asarray(self.keras_model_predict.predict_generator(
                prediction_generator, steps=data_gen_shape // self.batch_size))

            if result.ndim < 2:  # in case only 1 test data point, in such case we need to add a dimension
                result = np.expand_dims(result, axis=0)

            half_first_dim = result.shape[1] // 2  # result.shape[1] is guarantee an even number, otherwise sth is wrong

            predictions[i, :data_gen_shape] = result[:, :half_first_dim].reshape((data_gen_shape, self.labels_shape))
            predictions_var[i, :data_gen_shape] = result[:, half_first_dim:].reshape((data_gen_shape, self.labels_shape))

            if remainder_shape != 0:
                remainder_data = input_array[data_gen_shape:]
                remainder_data_err = inputs_err[data_gen_shape:]
                # assume its caused by mono images, so need to expand dim by 1
                if len(input_array[0].shape) != len(self.input_shape):
                    remainder_data = np.expand_dims(remainder_data, axis=-1)
                    remainder_data_err = np.expand_dims(remainder_data_err, axis=-1)
                result = self.keras_model_predict.predict({'input': remainder_data, 'input_err': remainder_data_err})
                predictions[i, data_gen_shape:] = result[:, :half_first_dim].reshape((remainder_shape, self.labels_shape))
                predictions_var[i, data_gen_shape:] = result[:, half_first_dim:].reshape((remainder_shape, self.labels_shape))

        print(f'Completed Dropout Variational Inference, {(time.time() - start_time):.{2}f}s in total')

        if self.labels_normalizer is not None:
            predictions = self.labels_normalizer.denormalize(predictions)
        else:
            predictions *= self.labels_std
            predictions += self.labels_mean

        pred = np.mean(predictions, axis=0)

        if self.task == 'regression':
            # Predictive variance
            mc_dropout_uncertainty = np.var(predictions, axis=0)  # var
            predictive_uncertainty = np.mean(np.exp(predictions_var) * (np.array(self.labels_std) ** 2), axis=0)
            pred_var = predictive_uncertainty + mc_dropout_uncertainty  # epistemic plus aleatoric uncertainty
            pred_uncertainty = np.sqrt(pred_var)  # Convert back to std error

            # final correction from variance to standard derivation
            mc_dropout_uncertainty = np.sqrt(mc_dropout_uncertainty)
            predictive_uncertainty = np.sqrt(predictive_uncertainty)

        elif self.task == 'classification':
            # we want entropy for classification uncertainty
            pred = np.argmax(pred, axis=1)
            predictions_var = np.mean(predictions_var, axis=0)
            mc_dropout_uncertainty = np.ones_like(pred, dtype=float)
            predictive_uncertainty = np.ones_like(pred, dtype=float)
            for i in range(pred.shape[0]):
                all_prediction = np.array(predictions[:, i, pred[i]])
                mc_dropout_uncertainty[i] = - np.sum(all_prediction * np.log(all_prediction))
                predictive_uncertainty[i] = np.array(predictions_var[i, pred[i]])

            pred_uncertainty = mc_dropout_uncertainty + predictive_uncertainty

        elif self.task == 'binary_classification':
            # we want entropy for classification uncertainty
            mc_dropout_uncertainty = - np.sum(pred * np.log(pred), axis=0)  # need to use raw prediction for uncertainty
            pred = np.rint(pred)
            predictive_uncertainty = np.mean(predictions_var, axis=0)
            pred_uncertainty = mc_dropout_uncertainty + predictive_uncertainty
        else:
            raise AttributeError('Unknown Task')

        return pred, {'total': pred_uncertainty, 'model': mc_dropout_uncertainty, 'predictive': predictive_uncertainty}

    def pre_training_checklist_child(self, input_data, labels, input_err, labels_err):
        self.pre_training_checklist_master(input_data, labels)

        if isinstance(input_data, H5Loader):
            self.targetname = input_data.target
            input_data, labels = input_data.load()

        # check if exists (exists mean fine-tuning, so we do not need calculate mean/std again)
        if self.input_normalizer is None:
            self.input_normalizer = Normalizer(mode=self.input_norm_mode)
            self.labels_normalizer = Normalizer(mode=self.labels_norm_mode)

            norm_data = self.input_normalizer.normalize(input_data)
            self.input_mean, self.input_std = self.input_normalizer.mean_labels, self.input_normalizer.std_labels
            norm_labels = self.labels_normalizer.normalize(labels)
            self.labels_mean, self.labels_std = self.labels_normalizer.mean_labels, self.labels_normalizer.std_labels
        else:
            norm_data = self.input_normalizer.normalize(input_data, calc=False)
            norm_labels = self.labels_normalizer.normalize(labels, calc=False)

        # No need to care about Magic number as loss function looks for magic num in y_true only
        norm_input_err = input_err / self.input_std
        norm_labels_err = labels_err / self.labels_std

        if self.keras_model is None:  # only compiler if there is no keras_model, e.g. fine-tuning does not required
            self.compile()

        self.train_idx, self.val_idx = train_test_split(np.arange(self.num_train), test_size=self.val_size)

        self.inv_model_precision = (2 * self.num_train * self.l2) / (self.length_scale ** 2 * (1 - self.dropout_rate))

        self.training_generator = BayesianCNNDataGenerator(self.batch_size).generate(norm_data[self.train_idx],
                                                                                     norm_labels[self.train_idx],
                                                                                     norm_input_err[self.train_idx],
                                                                                     norm_labels_err[self.train_idx])
        self.validation_generator = BayesianCNNDataGenerator(self.batch_size).generate(norm_data[self.val_idx],
                                                                                       norm_labels[self.val_idx],
                                                                                       norm_input_err[self.val_idx],
                                                                                       norm_labels_err[self.val_idx])

        return norm_data, norm_labels, norm_labels_err

    def post_training_checklist_child(self):
        astronn_model = 'model_weights.h5'
        self.keras_model.save(self.fullfilepath + astronn_model)
        print(astronn_model + f' saved to {(self.fullfilepath + astronn_model)}')

        self.hyper_txt.write(f"Dropout Rate: {self.dropout_rate} \n")
        self.hyper_txt.flush()
        self.hyper_txt.close()

        data = {'id': self.__class__.__name__ if self._model_identifier is None else self._model_identifier,
                'pool_length': self.pool_length, 'filterlen': self.filter_len,
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

        reduce_lr = ReduceLROnPlateau(monitor='val_output_loss', factor=0.5, min_delta=self.reduce_lr_epsilon,
                                      patience=self.reduce_lr_patience, min_lr=self.reduce_lr_min, mode='min',
                                      verbose=2)

        self.virtual_cvslogger = VirutalCSVLogger()

        self.__callbacks = [reduce_lr, self.virtual_cvslogger]  # default must have unchangeable callbacks

        if self.callbacks is not None:
            self.__callbacks.append(self.callbacks)

        start_time = time.time()

        self.history = self.keras_model.fit_generator(generator=self.training_generator,
                                                      steps_per_epoch=self.num_train // self.batch_size,
                                                      validation_data=self.validation_generator,
                                                      validation_steps=self.val_num // self.batch_size,
                                                      epochs=self.max_epochs, verbose=self.verbose,
                                                      workers=os.cpu_count(),
                                                      callbacks=self.__callbacks,
                                                      use_multiprocessing=MULTIPROCESS_FLAG)

        print(f'Completed Training, {(time.time() - start_time):.{2}f}s in total')

        if self.autosave is True:
            # Call the post training checklist to save parameters
            self.save()

        return None
