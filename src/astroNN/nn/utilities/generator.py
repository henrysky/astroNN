import numpy as np

import keras


class GeneratorBase(keras.utils.PyDataset):
    """
    Top-level data generator class to generate batches

    Subclass this class to create a custom data generator, need to implement __getitem__ method

    Parameters
    ----------
    batch_size: int
        batch size
    shuffle: bool
        shuffle the data or not after each epoch
    steps_per_epoch: int
        steps per epoch
    data: dict
        data dictionary
    np_rng: numpy.random.Generator
        numpy random generator

    History
    -------
    2019-Feb-17 - Updated - Henry Leung (University of Toronto)
    2024-Sept-6 - Updated - Henry Leung (University of Toronto)
    """

    def __init__(self, data, *, batch_size=32, shuffle=True, steps_per_epoch=None, np_rng=None, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.data = data
        self.shuffle = shuffle
        
        if steps_per_epoch is None:  # all data should shae the same length
            self.steps_per_epoch = int(np.ceil(len(data[list(data.keys())[0]]) / batch_size))
        else:
            self.steps_per_epoch = steps_per_epoch

        if np_rng is None:
            self.np_rng = np.random.default_rng()
        else:
            self.np_rng = np_rng

    def __len__(self):
        return self.steps_per_epoch

    def _get_exploration_order(self, idx_list):
        # shuffle (if applicable) and find exploration order
        if self.shuffle:
            idx_list = np.copy(idx_list)
            self.np_rng.shuffle(idx_list)

        return idx_list

    def input_d_checking(self, inputs, idx_list_temp):
        x_dict = {}
        float_dtype = keras.backend.floatx()
        for name in inputs.keys():
            if inputs[name].ndim == 2:
                x = np.empty(
                    (len(idx_list_temp), inputs[name].shape[1], 1),
                    dtype=float_dtype,
                )
                # Generate data
                x[:, :, 0] = inputs[name][idx_list_temp]

            elif inputs[name].ndim == 3:
                x = np.empty(
                    (
                        len(idx_list_temp),
                        inputs[name].shape[1],
                        inputs[name].shape[2],
                        1,
                    ),
                    dtype=float_dtype,
                )
                # Generate data
                x[:, :, :, 0] = inputs[name][idx_list_temp]

            elif inputs[name].ndim == 4:
                x = np.empty(
                    (
                        len(idx_list_temp),
                        inputs[name].shape[1],
                        inputs[name].shape[2],
                        inputs[name].shape[3],
                    ),
                    dtype=float_dtype,
                )
                # Generate data
                x[:, :, :, :] = inputs[name][idx_list_temp]
            else:
                raise ValueError(
                    f"Unsupported data dimension, your data has {inputs[name].ndim} dimension"
                )

            x_dict.update({name: x})

        return x_dict
