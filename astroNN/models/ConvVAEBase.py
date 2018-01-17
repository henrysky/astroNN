from abc import ABC, abstractmethod

import numpy as np
from keras.backend import clear_session
from keras.optimizers import Adam

from astroNN.datasets import H5Loader
from astroNN.models.NeuralNetMaster import NeuralNetMaster
from astroNN.models.utilities.generator import threadsafe_generator, GeneratorMaster
from astroNN.models.utilities.normalizer import Normalizer


class CVAE_DataGenerator(GeneratorMaster):
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

    def __init__(self, batch_size, shuffle=True):
        super(CVAE_DataGenerator, self).__init__(batch_size, shuffle)

    def _data_generation(self, spectra, list_IDs_temp):
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

                yield X, None


class ConvVAEBase(NeuralNetMaster, ABC):
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
        super(ConvVAEBase, self).__init__()
        self.name = 'Convolutional Variational Autoencoder'
        self._model_type = 'CVAE'
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
        self.latent_dim = None

        self.keras_vae = None
        self.keras_encoder = None
        self.keras_decoder = None

        self.input_shape = None

        self.input_normalizer = None
        self.recon_normalizer = None
        self.input_norm_mode = 1
        self.labels_norm_mode = 1
        self.input_mean_norm = None
        self.input_std_norm = None
        self.labels_mean_norm = None
        self.labels_std_norm = None

    def compile(self):
        self.keras_model, self.keras_vae, self.keras_encoder, self.keras_decoder = self.model()

        if self.optimizer is None or self.optimizer == 'adam':
            self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.optimizer_epsilon,
                                  decay=0.0)

        self.keras_model.compile(loss=None, optimizer=self.optimizer)
        return None

    @abstractmethod
    def train(self, input_data, input_recon_target):
        raise NotImplementedError

    def pre_training_checklist_child(self, input_data, input_recon_target):
        self.pre_training_checklist_master(input_data, input_recon_target)

        if isinstance(input_data, H5Loader):
            self.targetname = input_data.target
            input_data, input_recon_target = input_data.load()

        self.input_normalizer = Normalizer(mode=self.input_norm_mode)
        self.labels_normalizer = Normalizer(mode=self.labels_norm_mode)

        norm_data, self.input_mean_norm, self.input_std_norm = self.input_normalizer.normalize(input_data)
        norm_labels, self.labels_mean_norm, self.labels_std_norm = self.labels_normalizer.normalize(input_recon_target)

        self.input_shape = (norm_data.shape[1], 1,)
        self.labels_shape = norm_labels.shape[1]

        self.compile()
        self.plot_model()

        self.training_generator = CVAE_DataGenerator(self.batch_size).generate(norm_data, norm_labels)

        return input_data, input_recon_target

    def post_training_checklist_child(self):
        astronn_model = 'model_weights.h5'
        self.keras_model.save_weights(self.fullfilepath + astronn_model)
        print(astronn_model + ' saved to {}'.format(self.fullfilepath + astronn_model))

        np.savez(self.fullfilepath + '/astroNN_model_parameter.npz', id=self._model_identifier,
                 filterlen=self.filter_length,
                 filternum=self.num_filters, hidden=self.num_hidden, input=self.input_shape, labels=self.input_shape,
                 task=self.task, latent=self.latent_dim, input_mean=self.input_mean_norm,
                 labels_mean=self.labels_mean_norm, input_std=self.input_std_norm, labels_std=self.labels_std_norm,
                 valsize=self.val_size, targetname=self.targetname)

        clear_session()
