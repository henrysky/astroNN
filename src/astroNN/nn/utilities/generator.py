import numpy as np

import keras


class GeneratorBase(keras.utils.PyDataset):
    """
    Top-level data generator class to generate batches

    Subclass this class to create a custom data generator, need to implement __getitem__ method

    Parameters
    ----------
    data: dict
        data dictionary
    batch_size: int, optional (default is 64)
        batch size
    shuffle: bool, optional (default is True)
        shuffle the data or not after each epoch
    steps_per_epoch: int, optional (default is None)
        steps per epoch
    np_rng: numpy.random.Generator, optional (default is None)
        numpy random generator

    History
    -------
    2019-Feb-17 - Updated - Henry Leung (University of Toronto)
    2024-Sept-6 - Updated - Henry Leung (University of Toronto)
    """

    def __init__(self, data, *, batch_size=64, shuffle=True, steps_per_epoch=None, np_rng=None, **kwargs):
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
    
    def get_idx_item(self, data, idx):
        """
        Get batch data with index
        """
        if isinstance(data, dict):
            return {key: data[key][idx] for key in data.keys()}
        elif isinstance(data, list):
            return [data[i][idx] for i in range(len(data))]
        else:
            return data[idx]
