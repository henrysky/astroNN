import json
import os
from abc import ABC, abstractmethod

import numpy as np
from sklearn.model_selection import train_test_split

from astroNN.config import MULTIPROCESS_FLAG
from astroNN.config import keras_import_manager
from astroNN.datasets import H5Loader
from astroNN.models.NeuralNetMaster import NeuralNetMaster
from astroNN.nn.callbacks import VirutalCSVLogger
from astroNN.nn.utilities import Normalizer
from astroNN.nn.utilities.generator import threadsafe_generator, GeneratorMaster

keras = keras_import_manager()
regularizers = keras.regularizers
ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
Adam = keras.optimizers.Adam
Input = keras.layers.Input
Model = keras.models.Model


class CGANDataGenerator(GeneratorMaster):
    """
    NAME:
        CGANDataGenerator
    PURPOSE:
        To generate data for Keras
    INPUT:
    OUTPUT:
    HISTORY:
        2017-Dec-02 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, batch_size, shuffle=True):
        super().__init__(batch_size, shuffle)

    def _data_generation(self, inputs, recon_inputs, idx_list_temp):
        x = self.input_d_checking(inputs, idx_list_temp)
        y = self.input_d_checking(recon_inputs, idx_list_temp)

        return x, y

    @threadsafe_generator
    def generate(self, inputs, recon_inputs):
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
                x, y = self._data_generation(inputs, recon_inputs, idx_list_temp)

                yield x, y


class CGANPredDataGenerator(GeneratorMaster):
    """
    NAME:
        CGANPredDataGenerator
    PURPOSE:
        To generate data for Keras model prediction
    INPUT:
    OUTPUT:
    HISTORY:
        2017-Dec-02 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, batch_size, shuffle=False):
        super().__init__(batch_size, shuffle)

    def _data_generation(self, inputs, idx_list_temp):
        # Generate data
        x = self.input_d_checking(inputs, idx_list_temp)

        return x

    @threadsafe_generator
    def generate(self, inputs):
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
                x = self._data_generation(inputs, idx_list_temp)

                yield x


class CGANBase(NeuralNetMaster, ABC):
    """Top-level class for a Convolutional Variational Autoencoder"""

    def __init__(self):
        """
        NAME:
            __init__
        PURPOSE:
            To define astroNN Convolutional Variational Autoencoder
        HISTORY:
            2018-Jan-06 - Written - Henry Leung (University of Toronto)
        """
        super().__init__()
        self.name = 'Convolutional Generative Adversarial Network'
        self._model_type = 'CGAN'
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
        self.latent_dim = None
        self.val_size = 0.1
        self.dropout_rate = 0.0

        self.keras_vae = None
        self.keras_encoder = None
        self.keras_decoder = None

        self.input_shape = None

        self.input_normalizer = None
        self.recon_normalizer = None
        self.input_norm_mode = 255
        self.labels_norm_mode = 255
        self.input_mean = None
        self.input_std = None
        self.labels_mean = None
        self.labels_std = None

    @abstractmethod
    def discriminator(self):
        raise NotImplementedError

    @abstractmethod
    def generator(self):
        raise NotImplementedError

    def model(self):
        self.generator().compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4, decay=1e-5))
        self.discriminator().compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3, decay=1e-5))

        gan_input = Input(shape=[100])
        H = self.generator()(gan_input)
        gan_V = self.discriminator()(H)
        GAN = Model(gan_input, gan_V)
        return GAN

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None):
        self.keras_model = self.model()

        if optimizer is not None:
            self.optimizer = optimizer
        elif self.optimizer is None or self.optimizer == 'adam':
            self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.optimizer_epsilon,
                                  decay=0.0)

        self.keras_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4, decay=1e-5))
        return None

    def pre_training_checklist_child(self, input_data, input_recon_target):
        if self.task == 'classification':
            raise RuntimeError('astroNN VAE does not support classification task')

        self.pre_training_checklist_master(input_data, input_recon_target)

        if isinstance(input_data, H5Loader):
            self.targetname = input_data.target
            input_data, input_recon_target = input_data.load()

        self.input_normalizer = Normalizer(mode=self.input_norm_mode)
        self.labels_normalizer = Normalizer(mode=self.labels_norm_mode)

        # check if exists (exists mean fine-tuning, so we do not need calculate mean/std again)
        if self.input_mean is None:
            norm_data = self.input_normalizer.normalize(input_data)
            self.input_mean, self.input_std = self.input_normalizer.mean_labels, self.input_normalizer.std_labels
            norm_labels = self.labels_normalizer.normalize(input_recon_target)
            self.labels_mean, self.labels_std = self.labels_normalizer.mean_labels, self.labels_normalizer.std_labels
        else:
            norm_data = (input_data - self.input_mean) / self.input_std
            norm_labels = (input_recon_target - self.labels_mean) / self.labels_std

        if self.keras_model is None:  # only compiler if there is no keras_model, e.g. fine-tuning does not required
            self.compile()

        self.train_idx, self.val_idx = train_test_split(np.arange(self.num_train), test_size=self.val_size)

        self.training_generator = CGANDataGenerator(self.batch_size).generate(norm_data[self.train_idx],
                                                                              norm_labels[self.train_idx])
        self.validation_generator = CGANDataGenerator(self.batch_size).generate(norm_data[self.val_idx],
                                                                                norm_labels[self.val_idx])

        return input_data, input_recon_target

    def train(self, input_data, input_recon_target):
        # Call the checklist to create astroNN folder and save parameters
        self.pre_training_checklist_child(input_data, input_recon_target)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_delta=self.reduce_lr_epsilon,
                                      patience=self.reduce_lr_patience, min_lr=self.reduce_lr_min, mode='min',
                                      verbose=2)

        self.virtual_cvslogger = VirutalCSVLogger()

        self.keras_model.fit_generator(generator=self.training_generator,
                                       steps_per_epoch=self.num_train // self.batch_size,
                                       validation_data=self.validation_generator,
                                       validation_steps=self.val_num // self.batch_size,
                                       epochs=self.max_epochs, verbose=self.verbose, workers=os.cpu_count(),
                                       callbacks=[reduce_lr, self.virtual_cvslogger],
                                       use_multiprocessing=MULTIPROCESS_FLAG)

        if self.autosave is True:
            # Call the post training checklist to save parameters
            self.save()

        return None

    def test(self, input_data):
        self.pre_testing_checklist_master()
        # Prevent shallow copy issue
        input_array = np.array(input_data)
        input_array -= self.input_mean
        input_array /= self.input_std

        total_test_num = input_data.shape[0]  # Number of testing data

        # for number of training data smaller than batch_size
        if input_data.shape[0] < self.batch_size:
            self.batch_size = input_data.shape[0]

        # Due to the nature of how generator works, no overlapped prediction
        data_gen_shape = (total_test_num // self.batch_size) * self.batch_size
        remainder_shape = total_test_num - data_gen_shape  # Remainder from generator

        predictions = np.zeros((total_test_num, self.labels_shape, 1))

        # Data Generator for prediction
        prediction_generator = CGANPredDataGenerator(self.batch_size).generate(input_array[:data_gen_shape])
        predictions[:data_gen_shape] = np.asarray(self.keras_model.predict_generator(
            prediction_generator, steps=input_array.shape[0] // self.batch_size))

        if remainder_shape != 0:
            remainder_data = input_array[data_gen_shape:]
            # assume its caused by mono images, so need to expand dim by 1
            if len(input_array[0].shape) != len(self.input_shape):
                remainder_data = np.expand_dims(remainder_data, axis=-1)
            result = self.keras_model.predict(remainder_data)
            predictions[data_gen_shape:] = result

        if self.input_normalizer is not None:
            predictions[:, :, 0] = self.input_normalizer.denormalize(predictions[:, :, 0])
        else:
            predictions[:, :, 0] *= self.labels_std
            predictions[:, :, 0] += self.labels_mean

        return predictions

    def test_encoder(self, input_data):
        self.pre_testing_checklist_master()
        # Prevent shallow copy issue
        input_array = np.array(input_data)
        input_array -= self.input_mean
        input_array /= self.input_std

        total_test_num = input_data.shape[0]  # Number of testing data

        # for number of training data smaller than batch_size
        if input_data.shape[0] < self.batch_size:
            self.batch_size = input_data.shape[0]

        # Due to the nature of how generator works, no overlapped prediction
        data_gen_shape = (total_test_num // self.batch_size) * self.batch_size
        remainder_shape = total_test_num - data_gen_shape  # Remainder from generator

        encoding = np.zeros((total_test_num, self.latent_dim))

        # Data Generator for prediction
        prediction_generator = CGANPredDataGenerator(self.batch_size).generate(input_array[:data_gen_shape])
        encoding[:data_gen_shape] = np.asarray(self.keras_encoder.predict_generator(
            prediction_generator, steps=input_array.shape[0] // self.batch_size))

        if remainder_shape != 0:
            remainder_data = input_array[data_gen_shape:]
            # assume its caused by mono images, so need to expand dim by 1
            if len(input_array[0].shape) != len(self.input_shape):
                remainder_data = np.expand_dims(remainder_data, axis=-1)
            result = self.keras_encoder.predict(remainder_data)
            encoding[data_gen_shape:] = result

        return encoding

    def post_training_checklist_child(self):
        astronn_model = 'model_weights.h5'
        self.keras_model.save(self.fullfilepath + astronn_model)
        print(astronn_model + f' saved to {(self.fullfilepath + astronn_model)}')

        self.hyper_txt.write(f"Dropout Rate: {self.dropout_rate} \n")
        self.hyper_txt.flush()
        self.hyper_txt.close()

        np.savez(self.fullfilepath + '/astroNN_model_parameter.npz', id=self.__class__.__name__,
                 filterlen=self.filter_len,
                 filternum=self.num_filters, hidden=self.num_hidden, input=self.input_shape, labels=self.input_shape,
                 task=self.task, latent=self.latent_dim, input_mean=self.input_mean,
                 labels_mean=self.labels_mean, input_std=self.input_std, labels_std=self.labels_std,
                 valsize=self.val_size, targetname=self.targetname, l2=self.l2,
                 input_norm_mode=self.input_norm_mode, labels_norm_mode=self.labels_norm_mode,
                 batch_size=self.batch_size)

        data = {'id': self.__class__.__name__, 'pool_length': self.pool_length, 'filterlen': self.filter_len,
                'filternum': self.num_filters, 'hidden': self.num_hidden, 'input': self.input_shape,
                'labels': self.labels_shape, 'task': self.task, 'input_mean': self.input_mean.tolist(),
                'labels_mean': self.labels_mean.tolist(), 'input_std': self.input_std.tolist(),
                'labels_std': self.labels_std.tolist(),
                'valsize': self.val_size, 'targetname': self.targetname, 'dropout_rate': self.dropout_rate,
                'l2': self.l2, 'input_norm_mode': self.input_norm_mode, 'labels_norm_mode': self.labels_norm_mode,
                'batch_size': self.batch_size, 'latent': self.latent_dim}

        with open(self.fullfilepath + '/astroNN_model_parameter.json', 'w') as f:
            json.dump(data, f, indent=4, sort_keys=True)
