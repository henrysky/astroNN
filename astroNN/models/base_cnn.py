import json
import os
import time
from abc import ABC

import numpy as np
from tqdm import tqdm
from tensorflow import keras as tfk
from astroNN.config import MULTIPROCESS_FLAG
from astroNN.config import _astroNN_MODEL_NAME
from astroNN.models.base_master_nn import NeuralNetMaster
from astroNN.nn.callbacks import VirutalCSVLogger
from astroNN.nn.losses import categorical_crossentropy, binary_crossentropy
from astroNN.nn.losses import mean_squared_error, mean_absolute_error, mean_error
from astroNN.nn.metrics import categorical_accuracy, binary_accuracy
from astroNN.nn.utilities import Normalizer
from astroNN.nn.utilities.generator import GeneratorMaster
from astroNN.shared.dict_tools import dict_np_to_dict_list, list_to_dict
from astroNN.shared.warnings import deprecated, deprecated_copy_signature
from sklearn.model_selection import train_test_split

regularizers = tfk.regularizers
ReduceLROnPlateau, EarlyStopping = (
    tfk.callbacks.ReduceLROnPlateau,
    tfk.callbacks.EarlyStopping,
)
Adam = tfk.optimizers.Adam


class CNNDataGenerator(GeneratorMaster):
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
    :param sample_weight: Sample weights (if any)
    :type sample_weight: Union([NoneType, ndarray])
    :History:
        | 2017-Dec-02 - Written - Henry Leung (University of Toronto)
        | 2019-Feb-17 - Updated - Henry Leung (University of Toronto)
    """

    def __init__(
        self,
        batch_size,
        shuffle,
        steps_per_epoch,
        data,
        manual_reset=False,
        sample_weight=None,
    ):
        super().__init__(
            batch_size=batch_size,
            shuffle=shuffle,
            steps_per_epoch=steps_per_epoch,
            data=data,
            manual_reset=manual_reset,
        )
        self.inputs = self.data[0]
        self.labels = self.data[1]
        self.sample_weight = sample_weight

        # initial idx
        self.idx_list = self._get_exploration_order(
            range(self.inputs["input"].shape[0])
        )

    def _data_generation(self, idx_list_temp):
        x = self.input_d_checking(self.inputs, idx_list_temp)
        y = {}
        for name in self.labels.keys():
            y.update({name: self.labels[name][idx_list_temp]})
        if self.sample_weight is not None:
            return x, y, self.sample_weight[idx_list_temp]
        else:
            return x, y

    def __getitem__(self, index):
        return self._data_generation(
            self.idx_list[index * self.batch_size : (index + 1) * self.batch_size]
        )

    def on_epoch_end(self):
        # shuffle the list when epoch ends for the next epoch
        self.idx_list = self._get_exploration_order(
            range(self.inputs["input"].shape[0])
        )


class CNNPredDataGenerator(GeneratorMaster):
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
    :param pbar: tqdm progress bar
    :type pbar: obj
    :History:
        | 2017-Dec-02 - Written - Henry Leung (University of Toronto)
        | 2019-Feb-17 - Updated - Henry Leung (University of Toronto)
    """

    def __init__(
        self, batch_size, shuffle, steps_per_epoch, data, manual_reset=False, pbar=None
    ):
        super().__init__(
            batch_size=batch_size,
            shuffle=shuffle,
            steps_per_epoch=steps_per_epoch,
            data=data,
            manual_reset=manual_reset,
        )
        self.inputs = self.data[0]
        self.pbar = pbar

        # initial idx
        self.idx_list = self._get_exploration_order(
            range(self.inputs[list(self.inputs.keys())[0]].shape[0])
        )
        self.current_idx = -1

    def _data_generation(self, idx_list_temp):
        # Generate data
        x = self.input_d_checking(self.inputs, idx_list_temp)
        return x

    def __getitem__(self, index):
        x = self._data_generation(
            self.idx_list[index * self.batch_size : (index + 1) * self.batch_size]
        )
        if self.pbar and index > self.current_idx:
            self.pbar.update(self.batch_size)
        self.current_idx = index
        return x

    def on_epoch_end(self):
        # shuffle the list when epoch ends for the next epoch
        self.idx_list = self._get_exploration_order(
            range(self.inputs[list(self.inputs.keys())[0]].shape[0])
        )
        # reset counter


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
        super().__init__()
        self.name = "Convolutional Neural Network"
        self._model_type = "CNN"
        self._model_identifier = None
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
        self.dropout_rate = 0.0
        self.val_size = 0.1
        self.early_stopping_min_delta = 0.0001
        self.early_stopping_patience = 4

        self.input_norm_mode = 1
        self.labels_norm_mode = 2

    def compile(
        self,
        optimizer=None,
        loss=None,
        metrics=None,
        weighted_metrics=None,
        loss_weights=None,
        sample_weight_mode=None,
    ):
        if optimizer is not None:
            self.optimizer = optimizer
        elif self.optimizer is None or self.optimizer == "adam":
            self.optimizer = Adam(
                learning_rate=self.lr,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                epsilon=self.optimizer_epsilon,
            )
        if metrics is not None:
            self.metrics = metrics

        if self.task == "regression":
            self._last_layer_activation = "linear"
            loss_func = mean_squared_error if not loss else loss
            self.metrics = (
                [mean_absolute_error, mean_error] if not self.metrics else self.metrics
            )
        elif self.task == "classification":
            self._last_layer_activation = "softmax"
            loss_func = categorical_crossentropy if not loss else loss
            self.metrics = [categorical_accuracy] if not self.metrics else self.metrics
        elif self.task == "binary_classification":
            self._last_layer_activation = "sigmoid"
            loss_func = binary_crossentropy if not loss else loss
            self.metrics = [binary_accuracy] if not self.metrics else self.metrics
        else:
            raise RuntimeError(
                'Only "regression", "classification" and "binary_classification" are supported'
            )

        self.keras_model = self.model()

        self.keras_model.compile(
            loss=loss_func,
            optimizer=self.optimizer,
            metrics=self.metrics,
            weighted_metrics=weighted_metrics,
            loss_weights=loss_weights,
            sample_weight_mode=sample_weight_mode,
        )

        # inject custom training step if needed
        try:
            self.custom_train_step()
        except NotImplementedError:
            pass
        except TypeError:
            self.keras_model.train_step = self.custom_train_step
        # inject custom testing  step if needed
        try:
            self.custom_test_step()
        except NotImplementedError:
            pass
        except TypeError:
            self.keras_model.test_step = self.custom_test_step

        return None

    def recompile(
        self,
        loss=None,
        weighted_metrics=None,
        loss_weights=None,
        sample_weight_mode=None,
    ):
        """
        To be used when you need to recompile a already existing model
        """
        if self.task == "regression":
            self._last_layer_activation = "linear"
            loss_func = mean_squared_error if not loss else loss
            self.metrics = (
                [mean_absolute_error, mean_error] if not self.metrics else self.metrics
            )
        elif self.task == "classification":
            self._last_layer_activation = "softmax"
            loss_func = categorical_crossentropy if not loss else loss
            self.metrics = [categorical_accuracy] if not self.metrics else self.metrics
        elif self.task == "binary_classification":
            self._last_layer_activation = "sigmoid"
            loss_func = binary_crossentropy if not loss else loss
            self.metrics = [binary_accuracy] if not self.metrics else self.metrics
        else:
            raise RuntimeError(
                'Only "regression", "classification" and "binary_classification" are supported'
            )

    def pre_training_checklist_child(self, input_data, labels, sample_weight):
        # on top of checklist, convert input_data/labels to dict
        input_data, labels = self.pre_training_checklist_master(input_data, labels)

        # check if exists (existing means the model has already been trained (e.g. fine-tuning)
        # so we do not need calculate mean/std again)
        if self.input_normalizer is None:
            self.input_normalizer = Normalizer(
                mode=self.input_norm_mode, verbose=self.verbose
            )
            self.labels_normalizer = Normalizer(
                mode=self.labels_norm_mode, verbose=self.verbose
            )
            norm_data = self.input_normalizer.normalize(input_data)
            self.input_mean, self.input_std = (
                self.input_normalizer.mean_labels,
                self.input_normalizer.std_labels,
            )
            norm_labels = self.labels_normalizer.normalize(labels)
            self.labels_mean, self.labels_std = (
                self.labels_normalizer.mean_labels,
                self.labels_normalizer.std_labels,
            )
        else:
            norm_data = self.input_normalizer.normalize(input_data, calc=False)
            norm_labels = self.labels_normalizer.normalize(labels, calc=False)
        if (
            self.keras_model is None
        ):  # only compile if there is no keras_model, e.g. fine-tuning does not required
            self.compile()

        norm_data = self._tensor_dict_sanitize(norm_data, self.keras_model.input_names)
        norm_labels = self._tensor_dict_sanitize(
            norm_labels, self.keras_model.output_names
        )

        if self.has_val:
            self.train_idx, self.val_idx = train_test_split(
                np.arange(self.num_train + self.val_num), test_size=self.val_size
            )
        else:
            self.train_idx = np.arange(self.num_train + self.val_num)
            # just dummy, to minimize modification needed
            self.val_idx = np.arange(self.num_train + self.val_num)[:2]

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

        if sample_weight is not None:
            sample_weight_training = sample_weight[self.train_idx]
            sample_weight_val = sample_weight[self.val_idx]
        else:
            sample_weight_training = None
            sample_weight_val = None

        self.training_generator = CNNDataGenerator(
            batch_size=self.batch_size,
            shuffle=True,
            steps_per_epoch=self.num_train // self.batch_size,
            data=[norm_data_training, norm_labels_training],
            manual_reset=False,
            sample_weight=sample_weight_training,
        )

        if self.has_val:
            val_batchsize = (
                self.batch_size
                if len(self.val_idx) > self.batch_size
                else len(self.val_idx)
            )
            self.validation_generator = CNNDataGenerator(
                batch_size=val_batchsize,
                shuffle=False,
                steps_per_epoch=max(self.val_num // self.batch_size, 1),
                data=[norm_data_val, norm_labels_val],
                manual_reset=True,
                sample_weight=sample_weight_val,
            )

        return input_data, labels

    def fit(self, input_data, labels, sample_weight=None):
        """
        Train a Convolutional neural network

        :param input_data: Data to be trained with neural network
        :type input_data: ndarray
        :param labels: Labels to be trained with neural network
        :type labels: ndarray
        :param sample_weight: Sample weights (if any)
        :type sample_weight: Union([NoneType, ndarray])
        :return: None
        :rtype: NoneType
        :History: 2017-Dec-06 - Written - Henry Leung (University of Toronto)
        """
        # Call the checklist to create astroNN folder and save parameters
        self.pre_training_checklist_child(input_data, labels, sample_weight)

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            min_delta=self.reduce_lr_epsilon,
            patience=self.reduce_lr_patience,
            min_lr=self.reduce_lr_min,
            mode="min",
            verbose=self.verbose,
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            min_delta=self.early_stopping_min_delta,
            patience=self.early_stopping_patience,
            verbose=2,
            mode="min",
        )

        self.virtual_cvslogger = VirutalCSVLogger()

        self.__callbacks = [
            reduce_lr,
            self.virtual_cvslogger,
        ]  # default must have unchangeable callbacks

        if self.callbacks is not None:
            if isinstance(self.callbacks, list):
                self.__callbacks.extend(self.callbacks)
            else:
                self.__callbacks.append(self.callbacks)

        start_time = time.time()

        self.history = self.keras_model.fit(
            x=self.training_generator,
            validation_data=self.validation_generator,
            epochs=self.max_epochs,
            verbose=self.verbose,
            workers=os.cpu_count(),
            callbacks=self.__callbacks,
            use_multiprocessing=MULTIPROCESS_FLAG,
        )

        print(f"Completed Training, {(time.time() - start_time):.{2}f}s in total")

        if self.autosave is True:
            # Call the post training checklist to save parameters
            self.save()

        return None

    def fit_on_batch(self, input_data, labels, sample_weight=None):
        """
        Train a neural network by running a single gradient update on all of your data, suitable for fine-tuning

        :param input_data: Data to be trained with neural network
        :type input_data: ndarray
        :param labels: Labels to be trained with neural network
        :type labels: ndarray
        :param sample_weight: Sample weights (if any)
        :type sample_weight: Union([NoneType, ndarray])
        :return: None
        :rtype: NoneType
        :History: 2018-Aug-22 - Written - Henry Leung (University of Toronto)
        """

        input_data, labels = self.pre_training_checklist_master(input_data, labels)

        # check if exists (existing means the model has already been trained (e.g. fine-tuning),
        # so we do not need calculate mean/std again)
        if self.input_normalizer is None:
            self.input_normalizer = Normalizer(
                mode=self.input_norm_mode, verbose=self.verbose
            )
            self.labels_normalizer = Normalizer(
                mode=self.labels_norm_mode, verbose=self.verbose
            )

            norm_data = self.input_normalizer.normalize(input_data)
            self.input_mean, self.input_std = (
                self.input_normalizer.mean_labels,
                self.input_normalizer.std_labels,
            )
            norm_labels = self.labels_normalizer.normalize(labels)
            self.labels_mean, self.labels_std = (
                self.labels_normalizer.mean_labels,
                self.labels_normalizer.std_labels,
            )
        else:
            norm_data = self.input_normalizer.normalize(input_data, calc=False)
            norm_labels = self.labels_normalizer.normalize(labels, calc=False)

        start_time = time.time()

        fit_generator = CNNDataGenerator(
            batch_size=input_data["input"].shape[0],
            shuffle=False,
            steps_per_epoch=1,
            data=[norm_data, norm_labels],
            sample_weight=sample_weight,
        )

        scores = self.keras_model.fit(
            x=fit_generator,
            epochs=1,
            verbose=self.verbose,
            workers=os.cpu_count(),
            use_multiprocessing=MULTIPROCESS_FLAG,
        )

        print(
            f"Completed Training on Batch, {(time.time() - start_time):.{2}f}s in total"
        )

        return None

    def post_training_checklist_child(self):
        self.keras_model.save(self.fullfilepath + _astroNN_MODEL_NAME)
        print(
            _astroNN_MODEL_NAME
            + f" saved to {(self.fullfilepath + _astroNN_MODEL_NAME)}"
        )

        self.hyper_txt.write(f"Dropout Rate: {self.dropout_rate} \n")
        self.hyper_txt.flush()
        self.hyper_txt.close()

        data = {
            "id": self.__class__.__name__
            if self._model_identifier is None
            else self._model_identifier,
            "pool_length": self.pool_length,
            "filterlen": self.filter_len,
            "filternum": self.num_filters,
            "hidden": self.num_hidden,
            "input": self._input_shape,
            "labels": self._labels_shape,
            "task": self.task,
            "last_layer_activation": self._last_layer_activation,
            "activation": self.activation,
            "input_mean": dict_np_to_dict_list(self.input_mean),
            "labels_mean": dict_np_to_dict_list(self.labels_mean),
            "input_std": dict_np_to_dict_list(self.input_std),
            "labels_std": dict_np_to_dict_list(self.labels_std),
            "valsize": self.val_size,
            "targetname": self.targetname,
            "dropout_rate": self.dropout_rate,
            "l1": self.l1,
            "l2": self.l2,
            "maxnorm": self.maxnorm,
            "input_norm_mode": self.input_normalizer.normalization_mode,
            "labels_norm_mode": self.labels_normalizer.normalization_mode,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "batch_size": self.batch_size,
        }

        with open(self.fullfilepath + "/astroNN_model_parameter.json", "w") as f:
            json.dump(data, f, indent=4, sort_keys=True)

    def predict(self, input_data):
        """
        Use the neural network to do inference

        :param input_data: Data to be inferred with neural network
        :type input_data: ndarray
        :return: prediction and prediction uncertainty
        :rtype: ndarry
        :History: 2017-Dec-06 - Written - Henry Leung (University of Toronto)
        """
        self.has_model_check()
        input_data = self.pre_testing_checklist_master(input_data)

        input_array = self.input_normalizer.normalize(input_data, calc=False)
        total_test_num = input_data["input"].shape[0]  # Number of testing data

        # for number of training data smaller than batch_size
        if total_test_num < self.batch_size:
            self.batch_size = total_test_num

        # Due to the nature of how generator works, no overlapped prediction
        data_gen_shape = (total_test_num // self.batch_size) * self.batch_size
        remainder_shape = total_test_num - data_gen_shape  # Remainder from generator

        # TODO: named output????
        predictions = np.zeros((total_test_num, self._labels_shape["output"]))

        norm_data_main = {}
        norm_data_remainder = {}
        for name in input_array.keys():
            norm_data_main.update({name: input_array[name][:data_gen_shape]})
            norm_data_remainder.update({name: input_array[name][data_gen_shape:]})

        norm_data_main = self._tensor_dict_sanitize(
            norm_data_main, self.keras_model.input_names
        )
        norm_data_remainder = self._tensor_dict_sanitize(
            norm_data_remainder, self.keras_model.input_names
        )

        # Data Generator for prediction
        with tqdm(total=total_test_num, unit="sample") as pbar:
            pbar.set_description_str("Prediction progress: ")
            prediction_generator = CNNPredDataGenerator(
                batch_size=self.batch_size,
                shuffle=False,
                steps_per_epoch=total_test_num // self.batch_size,
                data=[norm_data_main],
                pbar=pbar,
            )
            predictions[:data_gen_shape] = np.asarray(
                self.keras_model.predict(prediction_generator, verbose=0)
            )

            if remainder_shape != 0:
                remainder_generator = CNNPredDataGenerator(
                    batch_size=remainder_shape,
                    shuffle=False,
                    steps_per_epoch=1,
                    data=[norm_data_remainder],
                )
                pbar.update(remainder_shape)
                predictions[data_gen_shape:] = np.asarray(
                    self.keras_model.predict(remainder_generator, verbose=0)
                )

        if self.labels_normalizer is not None:
            predictions = self.labels_normalizer.denormalize(
                list_to_dict(self.keras_model.output_names, predictions)
            )
        else:
            predictions *= self.labels_std
            predictions += self.labels_mean

        return predictions["output"]

    def evaluate(self, input_data, labels):
        """
        Evaluate neural network by provided input data and labels and get back a metrics score

        :param input_data: Data to be inferred with neural network
        :type input_data: ndarray
        :param labels: labels
        :type labels: ndarray
        :return: metrics score dictionary
        :rtype: dict
        :History: 2018-May-20 - Written - Henry Leung (University of Toronto)
        """
        self.has_model_check()
        input_data = list_to_dict(self.keras_model.input_names, input_data)
        labels = list_to_dict(self.keras_model.output_names, labels)

        # check if exists (existing means the model has already been trained (e.g. fine-tuning), so we do not need calculate mean/std again)
        if self.input_normalizer is None:
            self.input_normalizer = Normalizer(
                mode=self.input_norm_mode, verbose=self.verbose
            )
            self.labels_normalizer = Normalizer(
                mode=self.labels_norm_mode, verbose=self.verbose
            )

            norm_data = self.input_normalizer.normalize(input_data)
            self.input_mean, self.input_std = (
                self.input_normalizer.mean_labels,
                self.input_normalizer.std_labels,
            )
            norm_labels = self.labels_normalizer.normalize(labels)
            self.labels_mean, self.labels_std = (
                self.labels_normalizer.mean_labels,
                self.labels_normalizer.std_labels,
            )
        else:
            norm_data = self.input_normalizer.normalize(input_data, calc=False)
            norm_labels = self.labels_normalizer.normalize(labels, calc=False)

        norm_data = self._tensor_dict_sanitize(norm_data, self.keras_model.input_names)
        norm_labels = self._tensor_dict_sanitize(
            norm_labels, self.keras_model.output_names
        )

        total_num = input_data["input"].shape[0]
        eval_batchsize = self.batch_size if total_num > self.batch_size else total_num
        steps = total_num // self.batch_size if total_num > self.batch_size else 1

        start_time = time.time()
        print("Starting Evaluation")

        evaluate_generator = CNNDataGenerator(
            batch_size=eval_batchsize,
            shuffle=False,
            steps_per_epoch=steps,
            data=[norm_data, norm_labels],
        )

        scores = self.keras_model.evaluate(evaluate_generator)
        if isinstance(scores, float):  # make sure scores is iterable
            scores = list(str(scores))
        outputname = self.keras_model.output_names
        funcname = self.keras_model.metrics_names

        print(f"Completed Evaluation, {(time.time() - start_time):.{2}f}s elapsed")

        return list_to_dict(funcname, scores)

    @deprecated_copy_signature(fit)
    def train(self, *args, **kwargs):
        return self.fit(*args, **kwargs)

    @deprecated_copy_signature(fit_on_batch)
    def train_on_batch(self, *args, **kwargs):
        return self.fit_on_batch(*args, **kwargs)

    @deprecated_copy_signature(predict)
    def test(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
