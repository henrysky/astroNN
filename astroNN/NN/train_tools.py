# ---------------------------------------------------------#
#   astroNN.NN.train_tools: Tools to train models
# ---------------------------------------------------------#

import numpy as np
import h5py
import random


def load_batch_from_h5(data, num_train, batch_size, indx, mu_std, target=None):
    # Generate list of random indices (within the relevant partition of the main data file, e.g. the
    # training set) to be used to index into data_file
    indices = random.sample(range(indx, indx + num_train), batch_size)
    indices = np.sort(indices)

    mean_labels = mu_std[0]
    std_labels = mu_std[1]

    # load data
    X = data['spectra']
    X = X[indices, :]
    y = np.array((X.shape[1]))
    i = 0
    for tg in target:
        temp = data['{}'.format(tg)]
        if i == 0:
            y = temp[:][indices]
            i += 1
        else:
            y = np.column_stack((y, temp[:][indices]))

    # Normalize labels
    normed_y = (y - mean_labels) / std_labels

    # Reshape X data for compatibility with CNN
    X = X.reshape(len(X), X.shape[1], 1)

    return X, normed_y


def generate_train_batch(data_file, num_objects, batch_size, indx, mu_std, target):
    while True:
        x_batch, y_batch = load_batch_from_h5(data_file, num_objects, batch_size, indx, mu_std, target)
        yield (x_batch, y_batch)


def generate_cv_batch(data_file, num_objects, batch_size, indx, mu_std, target):
    while True:
        x_batch, y_batch = load_batch_from_h5(data_file, num_objects, batch_size, indx, mu_std, target)
        yield (x_batch, y_batch)
