# ---------------------------------------------------------#
#   astroNN.NN.test: test models
# ---------------------------------------------------------#

import tensorflow as tf
import h5py
import pylab as plt
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
import time
from matplotlib import gridspec
import seaborn as sns

import scipy


def apogee_test(model=None, testdata=None, folder_name=None):
    """
    NAME: apogee_test
    PURPOSE: To test the model and generate plots
    INPUT:
    OUTPUT: target and normalized data
    HISTORY:
        2017-Oct-14 Henry Leung
    """

    # prevent Tensorflow taking up all the GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    mean_and_std = np.load(folder_name + '\\meanstd_starnet.npy')
    target = np.load(folder_name + '\\targetname.npy')
    mean_labels = mean_and_std[0]
    std_labels = mean_and_std[1]
    num_labels = mean_and_std.shape[1]

    def denormalize(lb_norm):
        return ((lb_norm * std_labels) + mean_labels)

    def get_data(filename):
        i = 0
        f = h5py.File(testdata, 'r')
        spectra_array = f['spectra']
        labels_array = np.array((spectra_array.shape[1]))
        for tg in target:
            temp = f['{}'.format(tg)]
            if i == 0:
                labels_array = temp[:]
                i += 1
            else:
                labels_array = np.column_stack((labels_array, temp[:]))
        snr_array = f['SNR'][:]
        return (snr_array, spectra_array, labels_array)

    test_snr, test_spectra, test_labels = get_data(testdata)
    print('Test set contains ' + str(len(test_spectra)) + ' stars')
    model = load_model(model)

    def batch_predictions(model, spectra, batch_size, denormalize):
        predictions = np.zeros((len(spectra), num_labels))
        for i in range(len(spectra) // batch_size):
            inputs = spectra[i * batch_size:(i + 1) * batch_size].reshape((batch_size, spectra.shape[1], 1))
            predictions[i * batch_size:(i + 1) * batch_size] = denormalize(model.predict(inputs))
        inputs = spectra[(i + 1) * batch_size:].reshape((spectra[(i + 1) * batch_size:].shape[0], spectra.shape[1], 1))
        predictions[(i + 1) * batch_size:] = denormalize(model.predict(inputs))
        return predictions

    time1 = time.time()
    test_predictions = batch_predictions(model, test_spectra, 500, denormalize)
    print("{0:.2f}".format(time.time() - time1) + ' seconds to make ' + str(len(test_spectra)) + ' predictions')

    resid = test_predictions - test_labels
    bias = np.median(resid, axis=0)
    scatter = np.std(resid, axis=0)

    # Some plotting variables for asthetics
    plt.rcParams['axes.facecolor'] = 'white'
    sns.set_style("ticks")
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.alpha'] = '0.4'

    x_lab = 'ASPCAP'
    y_lab = 'astroNN'
    for i in range(num_labels):
        plt.figure(figsize=(10, 7), dpi=150)
        plt.scatter(test_predictions[:, i], resid[:, i], s=3)
        plt.xlabel('ASPCAP ' + target[i], fontsize=15)
        if i == 1:
            plt.ylabel('$\Delta$ ' + target[i] + '\n(' + y_lab + ' - ' + x_lab + ')\n', fontsize=15)
        else:
            plt.ylabel('$\Delta$ ' + target[i] + '\n(' + y_lab + ' - ' + x_lab + ')', fontsize=15)
        plt.tick_params(labelsize=10, width=1, length=10)
        ranges = (np.max(test_predictions[:, i]) - np.min(test_predictions[:, i])) / 2
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=3)
        plt.figtext(1,1,'$\widetilde{m}$=' + '{0:.3f}'.format(bias[i]) + ' $s$=' + '{0:.3f}'.format(scatter[i]/std_labels[i]),
                         size=10, bbox=bbox_props)
        plt.ylim([-ranges, ranges])
        plt.savefig(folder_name + '{}_test.png'.format(target[i]))

    return None
