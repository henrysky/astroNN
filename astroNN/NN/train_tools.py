# ---------------------------------------------------------#
#   astroNN.NN.train_tools: Tools to train models
# ---------------------------------------------------------#

import numpy as np
import random


def load_batch(num_train, batch_size, indx, mu_std, spectra, y):
    # Generate list of random indices (within the relevant partition of the main data file, e.g. the
    # training set) to be used to index into data_file
    indices = random.sample(range(indx, indx + num_train), batch_size)
    indices = np.sort(indices)

    mean_labels = mu_std[0]
    std_labels = mu_std[1]

    # load data
    spectra = spectra[indices, :]
    y = y[:][indices]

    # Normalize labels
    normed_y = (y - mean_labels) / std_labels

    # Reshape X data for compatibility with CNN
    spectra = spectra.reshape(len(spectra), spectra.shape[1], 1)

    return spectra, normed_y


def generate_train_batch(num_objects, batch_size, indx, mu_std, spectra, y):
    while True:
        x_batch, y_batch = load_batch(num_objects, batch_size, indx, mu_std, spectra, y)
        yield (x_batch, y_batch)


def generate_cv_batch(num_objects, batch_size, indx, mu_std, spectra, y):
    while True:
        x_batch, y_batch = load_batch(num_objects, batch_size, indx, mu_std, spectra, y)
        yield (x_batch, y_batch)