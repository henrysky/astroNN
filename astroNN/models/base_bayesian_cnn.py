import json
import os
import time
import warnings
from abc import ABC

import numpy as np
import tensorflow.keras as tfk
from astroNN.config import MULTIPROCESS_FLAG
from astroNN.config import _astroNN_MODEL_NAME
from astroNN.datasets import H5Loader
from astroNN.models.base_master_nn import NeuralNetMaster
from astroNN.nn.callbacks import VirutalCSVLogger
from astroNN.nn.layers import FastMCInference
from astroNN.nn.losses import mean_absolute_error, mean_error
from astroNN.nn.metrics import categorical_accuracy, binary_accuracy
from astroNN.nn.numpy import sigmoid
from astroNN.nn.utilities import Normalizer
from astroNN.nn.utilities.generator import GeneratorMaster
from astroNN.shared.custom_warnings import deprecated
from astroNN.shared.nn_tools import gpu_availability
from astroNN.shared.dict_tools import dict_np_to_dict_list, list_to_dict
from sklearn.model_selection import train_test_split

regularizers = tfk.regularizers
ReduceLROnPlateau = tfk.callbacks.ReduceLROnPlateau
Adam = tfk.optimizers.Adam


class BayesianCNNDataGenerator(GeneratorMaster):
    """
    To generate data to NN

    :param batch_size: batch size
    :type batch_size: int
    :param shuffle: Whether to shuffle batches or not
    :type shuffle: bool
    :param data: List of data to NN
    :type data: list
    :param manual_reset: Whether need to reset the generator manually, usually it is handled by tensorflow
    :type manual_reset: bool
    :History:
        | 2017-Dec-02 - Written - Henry Leung (University of Toronto)
        | 2019-Feb-17 - Updated - Henry Leung (University of Toronto)
    """

    def __init__(self, batch_size, shuffle, steps_per_epoch, data, manual_reset=False):
        super().__init__(batch_size=batch_size, shuffle=shuffle, steps_per_epoch=steps_per_epoch, data=data,
                         manual_reset=manual_reset)
        self.inputs = self.data[0]
        self.labels = self.data[1]

        # initial idx
        self.idx_list = self._get_exploration_order(range(self.inputs['input'].shape[0]))
        self.current_idx = 0

    def _data_generation(self, inputs, labels, idx_list_temp):
        x = self.input_d_checking(inputs, idx_list_temp)
        x.update({"labels_err": np.squeeze(x["labels_err"])})
        y = {}
        for name in labels.keys():
            y.update({name: labels[name][idx_list_temp]})
        return x, y

    def __getitem__(self, index):
        x, y = self._data_generation(self.inputs,
                                     self.labels,
                                     self.idx_list[self.current_idx:self.current_idx + self.batch_size])
        self.current_idx += self.batch_size
        if (self.current_idx+self.batch_size >= self.steps_per_epoch*self.batch_size-1) and self.manual_reset:
            self.current_idx = 0
        return x, y

    def on_epoch_end(self):
        # shuffle the list when epoch ends for the next epoch
        self.idx_list = self._get_exploration_order(range(self.inputs['input'].shape[0]))
        # reset counter
        self.current_idx = 0


class BayesianCNNPredDataGenerator(GeneratorMaster):
    """
    To generate data to NN for prediction

    :param batch_size: batch size
    :type batch_size: int
    :param shuffle: Whether to shuffle batches or not
    :type shuffle: bool
    :param data: List of data to NN
    :type data: list
    :param manual_reset: Whether need to reset the generator manually, usually it is handled by tensorflow
    :type manual_reset: bool
    :History:
        | 2017-Dec-02 - Written - Henry Leung (University of Toronto)
        | 2019-Feb-17 - Updated - Henry Leung (University of Toronto)
    """

    def __init__(self, batch_size, shuffle, steps_per_epoch, data, manual_reset=False):
        super().__init__(batch_size=batch_size, shuffle=shuffle, steps_per_epoch=steps_per_epoch, data=data,
                         manual_reset=manual_reset)
        self.inputs = self.data[0]

        # initial idx
        self.idx_list = self._get_exploration_order(range(self.inputs[list(self.inputs.keys())[0]].shape[0]))
        self.current_idx = 0

    def _data_generation(self, inputs, idx_list_temp):
        # Generate data
        x = self.input_d_checking(inputs, idx_list_temp)
        return x

    def __getitem__(self, index):
        x = self._data_generation(self.inputs, self.idx_list[self.current_idx:self.current_idx + self.batch_size])
        self.current_idx += self.batch_size
        if (self.current_idx+self.batch_size >= self.steps_per_epoch*self.batch_size-1) and self.manual_reset:
            self.current_idx = 0
        return x

    def on_epoch_end(self):
        # shuffle the list when epoch ends for the next epoch
        self.idx_list = self._get_exploration_order(range(self.inputs[list(self.inputs.keys())[0]].shape[0]))
        # reset counter
        self.current_idx = 0


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
        self.l1 = None
        self.l2 = None
        self.maxnorm = None
        self.inv_model_precision = None  # inverse model precision
        self.dropout_rate = 0.2
        self.length_scale = 3  # prior length scale
        self.mc_num = 100  # increased to 100 due to high performance VI on GPU implemented on 14 April 2018 (Henry)
        self.val_size = 0.1
        self.disable_dropout = False

        self.input_norm_mode = 1
        self.labels_norm_mode = 2

        self.keras_model_predict = None

    def pre_training_checklist_child(self, input_data, labels):
        input_data, labels = self.pre_training_checklist_master(input_data, labels)

        # check if exists (existing means the model has already been trained (e.g. fine-tuning), so we do not need calculate mean/std again)
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
        norm_data.update({"input_err":  (input_data['input_err'] / self.input_std['input']),
                          "labels_err": labels['labels_err'] / self.labels_std['output']})
        norm_labels.update({"labels_err": labels['labels_err'] / self.labels_std['output'],
                            "variance_output": norm_labels['output']})

        if self.keras_model is None:  # only compile if there is no keras_model, e.g. fine-tuning does not required
            self.compile()

        self.train_idx, self.val_idx = train_test_split(np.arange(self.num_train + self.val_num),
                                                        test_size=self.val_size)

        norm_data_training = {}
        norm_data_val = {}
        norm_labels_training = {}
        norm_labels_val = {}
        for name in norm_data.keys():
            norm_data_training.update({name: norm_data[name][self.train_idx]})
            norm_data_val.update({name: norm_data[name][self.val_idx]})
        for name in norm_labels.keys():
            norm_labels_training.update({name: norm_labels[name][self.train_idx]})
            norm_labels_val.update({name: norm_labels[name][self.val_idx]})

        self.inv_model_precision = (2 * self.num_train * self.l2) / (self.length_scale ** 2 * (1 - self.dropout_rate))

        self.training_generator = BayesianCNNDataGenerator(batch_size=self.batch_size,
                                                           shuffle=True,
                                                           steps_per_epoch=self.num_train // self.batch_size,
                                                           data=[norm_data_training,
                                                                 norm_labels_training],
                                                           manual_reset=False)

        val_batchsize = self.batch_size if len(self.val_idx) > self.batch_size else len(self.val_idx)
        self.validation_generator = BayesianCNNDataGenerator(batch_size=val_batchsize,
                                                             shuffle=False,
                                                             steps_per_epoch=max(self.val_num // self.batch_size, 1),
                                                             data=[norm_data_val,
                                                                   norm_labels_val],
                                                             manual_reset=True)

        return norm_data, norm_labels

    def compile(self, optimizer=None,
                loss=None,
                metrics=None,
                weighted_metrics=None,
                loss_weights=None,
                sample_weight_mode=None):
        if optimizer is not None:
            self.optimizer = optimizer
        elif self.optimizer is None or self.optimizer == 'adam':
            self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.optimizer_epsilon,
                                  decay=0.0)
        if metrics is not None:
            self.metrics = metrics
        if self.task == 'regression':
            if self._last_layer_activation is None:
                self._last_layer_activation = 'linear'
        elif self.task == 'classification':
            if self._last_layer_activation is None:
                self._last_layer_activation = 'softmax'
        elif self.task == 'binary_classification':
            if self._last_layer_activation is None:
                self._last_layer_activation = 'sigmoid'
        else:
            raise RuntimeError('Only "regression", "classification" and "binary_classification" are supported')

        self.keras_model, self.keras_model_predict, output_loss, variance_loss = self.model()

        if self.task == 'regression':
            self.metrics = [mean_absolute_error, mean_error] if not self.metrics else self.metrics
            self.keras_model.compile(loss={'output': output_loss, 'variance_output': variance_loss},
                                     optimizer=self.optimizer,
                                     metrics={'output': self.metrics},
                                     weighted_metrics=weighted_metrics,
                                     loss_weights={'output': .5,
                                                   'variance_output': .5} if not loss_weights else loss_weights,
                                     sample_weight_mode=sample_weight_mode)
        elif self.task == 'classification':
            self.metrics = [categorical_accuracy] if not self.metrics else self.metrics
            self.keras_model.compile(loss={'output': output_loss, 'variance_output': variance_loss},
                                     optimizer=self.optimizer,
                                     metrics={'output': self.metrics},
                                     weighted_metrics=weighted_metrics,
                                     loss_weights={'output': .5,
                                                   'variance_output': .5} if not loss_weights else loss_weights,
                                     sample_weight_mode=sample_weight_mode)
        elif self.task == 'binary_classification':
            self.metrics = [binary_accuracy] if not self.metrics else self.metrics
            self.keras_model.compile(loss={'output': output_loss, 'variance_output': variance_loss},
                                     optimizer=self.optimizer,
                                     metrics={'output': self.metrics},
                                     weighted_metrics=weighted_metrics,
                                     loss_weights={'output': .5,
                                                   'variance_output': .5} if not loss_weights else loss_weights,
                                     sample_weight_mode=sample_weight_mode)
        return None

    def train(self, input_data, labels, inputs_err=None, labels_err=None):
        """
        Train a Bayesian neural network

        :param input_data: Data to be trained with neural network
        :type input_data: ndarray
        :param labels: Labels to be trained with neural network
        :type labels: ndarray
        :param inputs_err: Error for input_data (if any), same shape with input_data.
        :type inputs_err: Union([NoneType, ndarray])
        :param labels_err: Labels error (if any)
        :type labels_err: Union([NoneType, ndarray])
        :return: None
        :rtype: NoneType
        :History:
            | 2018-Jan-06 - Written - Henry Leung (University of Toronto)
            | 2018-Apr-12 - Updated - Henry Leung (University of Toronto)
        """
        if inputs_err is None:
            inputs_err = np.zeros_like(input_data)

        if labels_err is None:
            labels_err = np.zeros_like(labels)

        # TODO: allow named inputs too??
        input_data = {"input": input_data, "input_err": inputs_err, "labels_err": labels_err}
        labels = {"output": labels, "labels_err": labels_err, "variance_output": labels}

        # Call the checklist to create astroNN folder and save parameters
        input_data, labels = self.pre_training_checklist_child(input_data, labels)

        reduce_lr = ReduceLROnPlateau(monitor='val_output_loss', factor=0.5, min_delta=self.reduce_lr_epsilon,
                                      patience=self.reduce_lr_patience, min_lr=self.reduce_lr_min, mode='min',
                                      verbose=2)

        self.virtual_cvslogger = VirutalCSVLogger()

        self.__callbacks = [reduce_lr, self.virtual_cvslogger]  # default must have unchangeable callbacks

        if self.callbacks is not None:
            if isinstance(self.callbacks, list):
                self.__callbacks.extend(self.callbacks)
            else:
                self.__callbacks.append(self.callbacks)

        start_time = time.time()

        self.history = self.keras_model.fit_generator(generator=self.training_generator,
                                                      validation_data=self.validation_generator,
                                                      epochs=self.max_epochs, verbose=self.verbose,
                                                      workers=os.cpu_count(),
                                                      callbacks=self.__callbacks,
                                                      use_multiprocessing=MULTIPROCESS_FLAG)

        print(f'Completed Training, {(time.time() - start_time):.{2}f}s in total')

        if self.autosave is True:
            # Call the post training checklist to save parameters
            self.save()

        return None

    def train_on_batch(self, input_data, labels, inputs_err=None, labels_err=None):
        """
        Train a Bayesian neural network by running a single gradient update on all of your data, suitable for fine-tuning

        :param input_data: Data to be trained with neural network
        :type input_data: ndarray
        :param labels: Labels to be trained with neural network
        :type labels: ndarray
        :param inputs_err: Error for input_data (if any), same shape with input_data.
        :type inputs_err: Union([NoneType, ndarray])
        :param labels_err: Labels error (if any)
        :type labels_err: Union([NoneType, ndarray])
        :return: None
        :rtype: NoneType
        :History:
            | 2018-Aug-25 - Written - Henry Leung (University of Toronto)
        """
        self.has_model_check()

        if inputs_err is None:
            inputs_err = np.zeros_like(input_data)

        if labels_err is None:
            labels_err = np.zeros_like(labels)

        # check if exists (existing means the model has already been trained (e.g. fine-tuning), so we do not need calculate mean/std again)
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
        norm_input_err = inputs_err / self.input_std['input']
        norm_labels_err = labels_err / self.labels_std['output']

        norm_data.update({"input_err": norm_input_err, "labels_err": norm_labels_err})
        norm_labels.update({"labels_err": norm_labels_err, "variance_output": norm_labels})

        start_time = time.time()

        fit_generator = BayesianCNNDataGenerator(batch_size=input_data.shape[0],
                                                 shuffle=False,
                                                 steps_per_epoch=1,
                                                 data=[norm_data,
                                                       norm_labels,
                                                       norm_input_err,
                                                       norm_labels_err])

        score = self.keras_model.fit_generator(fit_generator,
                                               epochs=1,
                                               verbose=self.verbose,
                                               workers=os.cpu_count(),
                                               use_multiprocessing=MULTIPROCESS_FLAG)

        print(f'Completed Training on Batch, {(time.time() - start_time):.{2}f}s in total')

        return None

    def post_training_checklist_child(self):
        self.keras_model.save(self.fullfilepath + _astroNN_MODEL_NAME)
        print(_astroNN_MODEL_NAME + f' saved to {(self.fullfilepath + _astroNN_MODEL_NAME)}')

        self.hyper_txt.write(f"Dropout Rate: {self.dropout_rate} \n")
        self.hyper_txt.flush()
        self.hyper_txt.close()

        data = {'id': self.__class__.__name__ if self._model_identifier is None else self._model_identifier,
                'pool_length': self.pool_length,
                'filterlen': self.filter_len,
                'filternum': self.num_filters,
                'hidden': self.num_hidden,
                'input': self._input_shape,
                'labels': self._labels_shape,
                'task': self.task,
                'last_layer_activation': self._last_layer_activation,
                'activation': self.activation,
                'input_mean': dict_np_to_dict_list(self.input_mean),
                'inv_tau': self.inv_model_precision,
                'length_scale': self.length_scale,
                'labels_mean': dict_np_to_dict_list(self.labels_mean),
                'input_std': dict_np_to_dict_list(self.input_std),
                'labels_std': dict_np_to_dict_list(self.labels_std),
                'valsize': self.val_size,
                'targetname': self.targetname,
                'dropout_rate': self.dropout_rate,
                'l1': self.l1,
                'l2': self.l2,
                'maxnorm': self.maxnorm,
                'input_norm_mode': self.input_norm_mode,
                'labels_norm_mode': self.labels_norm_mode,
                'input_names': self.input_names,
                'output_names': self.output_names,
                'batch_size': self.batch_size}

        with open(self.fullfilepath + '/astroNN_model_parameter.json', 'w') as f:
            json.dump(data, f, indent=4, sort_keys=True)

    def test(self, input_data, inputs_err=None):
        """
        Test model, High performance version designed for fast variational inference on GPU

        :param input_data: Data to be inferred with neural network
        :type input_data: ndarray
        :param inputs_err: Error for input_data, same shape with input_data.
        :type inputs_err: Union([NoneType, ndarray])
        :return: prediction and prediction uncertainty
        :History:
            | 2018-Jan-06 - Written - Henry Leung (University of Toronto)
            | 2018-Apr-12 - Updated - Henry Leung (University of Toronto)
        """
        self.has_model_check()

        if gpu_availability() is False and self.mc_num > 25:
            warnings.warn(f'You are using CPU version Tensorflow, doing {self.mc_num} times Monte Carlo Inference can '
                          f'potentially be very slow! \n '
                          f'A possible fix is to decrease the mc_num parameter of the model to do less MC Inference \n'
                          f'This is just a warning, and will not shown if mc_num < 25 on CPU')
            if self.mc_num < 2:
                raise AttributeError("mc_num cannot be smaller than 2")

                # if no error array then just zeros
        if inputs_err is None:
            inputs_err = np.zeros_like(input_data)
        else:
            inputs_err = np.atleast_2d(inputs_err)
            inputs_err /= self.input_std['input']

        input_data = {"input": input_data, "input_err": inputs_err}
        input_data = self.pre_testing_checklist_master(input_data)

        if self.input_normalizer is not None:
            input_array = self.input_normalizer.normalize(input_data, calc=False)
        else:
            # Prevent shallow copy issue
            input_array = np.array(input_data)
            input_array -= self.input_mean['input']
            input_array /= self.input_std['input']

        total_test_num = input_data['input'].shape[0]  # Number of testing data

        # for number of training data smaller than batch_size
        if total_test_num < self.batch_size:
            batch_size = total_test_num
        else:
            batch_size = self.batch_size

        # Due to the nature of how generator works, no overlapped prediction
        data_gen_shape = (total_test_num // batch_size) * batch_size
        remainder_shape = total_test_num - data_gen_shape  # Remainder from generator

        norm_data_main = {}
        norm_data_remainder = {}
        for name in input_array.keys():
            norm_data_main.update({name: input_array[name][:data_gen_shape]})
            norm_data_remainder.update({name: input_array[name][data_gen_shape:]})

        start_time = time.time()
        print("Starting Dropout Variational Inference")

        # Data Generator for prediction
        prediction_generator = BayesianCNNPredDataGenerator(batch_size=batch_size,
                                                            shuffle=False,
                                                            steps_per_epoch=data_gen_shape // batch_size,
                                                            data=[norm_data_main])

        new = FastMCInference(self.mc_num)(self.keras_model_predict)

        result = np.asarray(new.predict_generator(prediction_generator))

        if remainder_shape != 0:  # deal with remainder
            remainder_generator = BayesianCNNPredDataGenerator(batch_size=remainder_shape,
                                                               shuffle=False,
                                                               steps_per_epoch=1,
                                                               data=[norm_data_remainder])
            remainder_result = np.asarray(new.predict_generator(remainder_generator))
            if remainder_shape == 1:
                remainder_result = np.expand_dims(remainder_result, axis=0)
            result = np.concatenate((result, remainder_result))

        # in case only 1 test data point, in such case we need to add a dimension
        if result.ndim < 3 and batch_size == 1:
            result = np.expand_dims(result, axis=0)

        half_first_dim = result.shape[1] // 2  # result.shape[1] is guarantee an even number, otherwise sth is wrong

        predictions = result[:, :half_first_dim, 0]  # mean prediction
        mc_dropout_uncertainty = result[:, :half_first_dim, 1] * (self.labels_std['output'] ** 2)  # model uncertainty
        predictions_var = np.exp(result[:, half_first_dim:, 0]) * (self.labels_std['output'] ** 2)  # predictive uncertainty

        print(f'Completed Dropout Variational Inference with {self.mc_num} forward passes, '
              f'{(time.time() - start_time):.{2}f}s elapsed')

        if self.labels_normalizer is not None:
            predictions = self.labels_normalizer.denormalize(list_to_dict([self.keras_model.output_names[0]], predictions))
            predictions = predictions['output']
        else:
            predictions *= self.labels_std['output']
            predictions += self.labels_mean['output']

        if self.task == 'regression':
            # Predictive variance
            pred_var = predictions_var + mc_dropout_uncertainty  # epistemic plus aleatoric uncertainty
            pred_uncertainty = np.sqrt(pred_var)  # Convert back to std error

            # final correction from variance to standard derivation
            mc_dropout_uncertainty = np.sqrt(mc_dropout_uncertainty)
            predictive_uncertainty = np.sqrt(predictions_var)

        elif self.task == 'classification':
            # we want entropy for classification uncertainty
            predicted_class = np.argmax(predictions, axis=1)
            mc_dropout_uncertainty = np.ones_like(predicted_class, dtype=float)
            predictive_uncertainty = np.ones_like(predicted_class, dtype=float)

            # center variance
            predictions_var -= 1.
            for i in range(predicted_class.shape[0]):
                all_prediction = np.array(predictions[i, :])
                mc_dropout_uncertainty[i] = - np.sum(all_prediction * np.log(all_prediction))
                predictive_uncertainty[i] = predictions_var[i, predicted_class[i]]

            pred_uncertainty = mc_dropout_uncertainty + predictive_uncertainty
            # We only want the predicted class back
            predictions = predicted_class

        elif self.task == 'binary_classification':
            # we want entropy for classification uncertainty, so need prediction in logits space
            mc_dropout_uncertainty = - np.sum(predictions * np.log(predictions), axis=0)
            # need to activate before round to int so that the prediction is always 0 or 1
            predictions = np.rint(sigmoid(predictions))
            predictive_uncertainty = predictions_var
            pred_uncertainty = mc_dropout_uncertainty + predictions_var

        else:
            raise AttributeError('Unknown Task')

        return predictions, {'total': pred_uncertainty, 'model': mc_dropout_uncertainty,
                             'predictive': predictive_uncertainty}

    @deprecated
    def test_old(self, input_data, inputs_err=None):
        """
        Tests model, it is recommended to use the new test() instead of this deprecated method

        :param input_data: Data to be inferred with neural network
        :type input_data: ndarray
        :param inputs_err: Error for input_data, same shape with input_data.
        :type inputs_err: Union([NoneType, ndarray])
        :return: prediction and prediction uncertainty
        :History: 2018-Jan-06 - Written - Henry Leung (University of Toronto)
        """

        input_data = {"input": input_data, "input_err": inputs_err}
        input_data = self.pre_testing_checklist_master(input_data)

        if self.input_normalizer is not None:
            input_array = self.input_normalizer.normalize(input_data, calc=False)
        else:
            # Prevent shallow copy issue
            input_array = np.array(input_data)
            input_array -= self.input_mean['input']
            input_array /= self.input_std['input']

        # if no error array then just zeros
        if inputs_err is None:
            inputs_err = np.zeros_like(input_data)
        else:
            inputs_err /= self.input_std['input']

        total_test_num = input_data.shape[0]  # Number of testing data

        # for number of training data smaller than batch_size
        if total_test_num < self.batch_size:
            self.batch_size = total_test_num

        predictions = np.zeros((self.mc_num, total_test_num, self._labels_shape))
        predictions_var = np.zeros((self.mc_num, total_test_num, self._labels_shape))

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
            prediction_generator = BayesianCNNPredDataGenerator(batch_size=self.batch_size,
                                                                shuffle=False,
                                                                steps_per_epoch=data_gen_shape // self.batch_size,
                                                                data=[input_array[:data_gen_shape],
                                                                      inputs_err[:data_gen_shape]])

            result = np.asarray(self.keras_model_predict.predict_generator(prediction_generator))

            if result.ndim < 2:  # in case only 1 test data point, in such case we need to add a dimension
                result = np.expand_dims(result, axis=0)

            half_first_dim = result.shape[1] // 2  # result.shape[1] is guarantee an even number, otherwise sth is wrong

            predictions[i, :data_gen_shape] = result[:, :half_first_dim].reshape((data_gen_shape, self._labels_shape))
            predictions_var[i, :data_gen_shape] = result[:, half_first_dim:].reshape(
                (data_gen_shape, self._labels_shape))

            if remainder_shape != 0:
                remainder_data = input_array[data_gen_shape:]
                remainder_data_err = inputs_err[data_gen_shape:]
                # assume its caused by mono images, so need to expand dim by 1
                if len(input_array[0].shape) != len(self._input_shape):
                    remainder_data = np.expand_dims(remainder_data, axis=-1)
                    remainder_data_err = np.expand_dims(remainder_data_err, axis=-1)
                result = self.keras_model_predict.predict({'input': remainder_data, 'input_err': remainder_data_err})
                predictions[i, data_gen_shape:] = result[:, :half_first_dim].reshape(
                    (remainder_shape, self._labels_shape))
                predictions_var[i, data_gen_shape:] = result[:, half_first_dim:].reshape(
                    (remainder_shape, self._labels_shape))

        print(f'Completed Dropout Variational Inference, {(time.time() - start_time):.{2}f}s in total')

        if self.labels_normalizer is not None:
            predictions = self.labels_normalizer.denormalize(list_to_dict(self.keras_model.output_names, predictions))
        else:
            predictions *= self.labels_std['input']
            predictions += self.labels_mean['input']

        pred = np.mean(predictions, axis=0)

        if self.task == 'regression':
            # Predictive variance
            mc_dropout_uncertainty = np.var(predictions, axis=0)  # var
            predictive_uncertainty = np.mean(np.exp(predictions_var) * (np.array(self.labels_std['output']) ** 2), axis=0)
            pred_var = predictive_uncertainty + mc_dropout_uncertainty  # epistemic plus aleatoric uncertainty
            pred_uncertainty = np.sqrt(pred_var)  # Convert back to std error

            # final correction from variance to standard derivation
            mc_dropout_uncertainty = np.sqrt(mc_dropout_uncertainty)
            predictive_uncertainty = np.sqrt(predictive_uncertainty)

        elif self.task == 'classification':
            # we want entropy for classification uncertainty
            pred_shape = pred.shape[0]
            pred = np.argmax(pred, axis=1)
            predictions_var = np.mean(predictions_var, axis=0)
            mc_dropout_uncertainty = np.ones_like(pred, dtype=float)
            predictive_uncertainty = np.ones_like(pred, dtype=float)
            for i in range(pred_shape):
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

    def evaluate(self, input_data, labels, inputs_err=None, labels_err=None):
        """
        Evaluate neural network by provided input data and labels and get back a metrics score

        :param input_data: Data to be trained with neural network
        :type input_data: ndarray
        :param labels: Labels to be trained with neural network
        :type labels: ndarray
        :param inputs_err: Error for input_data (if any), same shape with input_data.
        :type inputs_err: Union([NoneType, ndarray])
        :param labels_err: Labels error (if any)
        :type labels_err: Union([NoneType, ndarray])
        :return: metrics score dictionary
        :rtype: dict
        :History: 2018-May-20 - Written - Henry Leung (University of Toronto)
        """
        self.has_model_check()

        if inputs_err is None:
            inputs_err = np.zeros_like(input_data)

        if labels_err is None:
            labels_err = np.zeros_like(labels)

        input_data = {"input": input_data}
        labels = {"output": labels}

        # check if exists (existing means the model has already been trained (e.g. fine-tuning), so we do not need calculate mean/std again)
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
        norm_input_err = inputs_err / self.input_std['input']
        norm_labels_err = labels_err / self.labels_std['output']

        norm_data.update({"input_err": norm_input_err, "labels_err": norm_labels_err})
        norm_labels.update({"labels_err": norm_labels_err, "variance_output": norm_labels["output"]})

        total_num = input_data['input'].shape[0]
        eval_batchsize = self.batch_size if total_num > self.batch_size else total_num
        steps = total_num // self.batch_size if total_num > self.batch_size else 1

        start_time = time.time()
        print("Starting Evaluation")

        evaluate_generator = BayesianCNNDataGenerator(batch_size=eval_batchsize,
                                                      shuffle=False,
                                                      steps_per_epoch=steps,
                                                      data=[norm_data,
                                                            norm_labels])

        scores = self.keras_model.evaluate_generator(evaluate_generator)
        if isinstance(scores, float):  # make sure scores is iterable
            scores = list(str(scores))
        outputname = self.keras_model.output_names
        funcname = []
        if isinstance(self.keras_model.metrics, dict):
            func_list = self.keras_model.metrics[outputname[0]]
        else:
            func_list = self.keras_model.metrics
        for func in func_list:
            if hasattr(func, __name__):
                funcname.append(func.__name__)
            else:
                funcname.append(func.__class__.__name__)
        # funcname = [func.__name__ for func in self.keras_model.metrics[outputname[0]]]
        loss_outputname = ['loss_' + name for name in outputname]
        output_funcname = [outputname[0] + '_' + name for name in funcname]
        list_names = ['loss', *loss_outputname, *output_funcname]

        print(f'Completed Evaluation, {(time.time() - start_time):.{2}f}s elapsed')

        return list_to_dict(list_names, scores)
