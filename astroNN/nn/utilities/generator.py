import numpy as np

from astroNN.config import keras_import_manager

keras = keras_import_manager()
Sequence = keras.utils.data_utils.Sequence


class GeneratorMaster(Sequence):
    """
    | Top-level class of astroNN data pipeline to generate data for NNs.
    | It is implemented based on Tensorflow data ``Sequence`` class.

    You need to implement the ``__getitem__`` in the generator sub-class

    :History: 2019-Feb-17 - Updated - Henry Leung (University of Toronto)
    """

    def __init__(self, batch_size, shuffle, steps_per_epoch, data):
        self.batch_size = batch_size
        self.data = data
        self.shuffle = shuffle

        self.steps_per_epoch = steps_per_epoch

    def __len__(self):
        return self.steps_per_epoch

    def _get_exploration_order(self, idx_list):
        """
        :param idx_list:
        :return:
        """
        # shuffle (if applicable) and find exploration order
        if self.shuffle is True:
            idx_list = np.copy(idx_list)
            np.random.shuffle(idx_list)

        return idx_list

    def sparsify(self, y):
        """Returns labels in binary NumPy array"""
        # n_classes =  # Enter number of classes
        # return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
        #                  for i in range(y.shape[0])])
        pass

    def input_d_checking(self, inputs, idx_list_temp):
        if inputs.ndim == 2:
            x = np.empty((len(idx_list_temp), inputs.shape[1], 1))
            # Generate data
            x[:, :, 0] = inputs[idx_list_temp]

        elif inputs.ndim == 3:
            x = np.empty((len(idx_list_temp), inputs.shape[1], inputs.shape[2], 1))
            # Generate data
            x[:, :, :, 0] = inputs[idx_list_temp]

        elif inputs.ndim == 4:
            x = np.empty((len(idx_list_temp), inputs.shape[1], inputs.shape[2], inputs.shape[3]))
            # Generate data
            x[:, :, :, :] = inputs[idx_list_temp]
        else:
            raise ValueError(f"Unsupported data dimension, your data has {inputs.ndim} dimension")

        return x
