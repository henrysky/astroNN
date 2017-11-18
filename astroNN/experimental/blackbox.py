# ---------------------------------------------------------#
#   astroNN.experimental.blackbox: eval NN attention via sliding a blackbox
# ---------------------------------------------------------#
import os
import time
from functools import reduce

import h5py
import numpy as np
import pylab as plt
import seaborn as sns
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from keras.models import load_model

from astroNN.NN.test import batch_predictions, target_name_conversion
from astroNN.shared.nn_tools import h5name_check


def blackbox_eval(h5name=None, folder_name=None):
    """
    NAME: blackbox_eval
    PURPOSE: To eval NN attention via sliding a blackbox
    INPUT:
        h5name = Name of the h5 data set
        folder_name = the folder name contains the model
    OUTPUT: plots
    HISTORY:
        2017-Nov-17 Henry Leung
    """

    # prevent Tensorflow taking up all the GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    h5name_check(h5name)

    traindata = h5name + '_train.h5'

    currentdir = os.getcwd()
    fullfolderpath = currentdir + '/' + folder_name
    print(fullfolderpath)
    mean_and_std = np.load(fullfolderpath + '/meanstd.npy')
    spec_meanstd = np.load(fullfolderpath + '/spectra_meanstd.npy')
    target = np.load(fullfolderpath + '/targetname.npy')
    modelname = '/model_{}.h5'.format(folder_name[-11:])
    model = load_model(os.path.normpath(fullfolderpath + modelname))

    mean_labels = mean_and_std[0]
    std_labels = mean_and_std[1]
    num_labels = mean_and_std.shape[1]

    # ensure the file will be cleaned up
    with h5py.File(traindata) as F:
        i = 0
        index_not9999 = []
        for tg in target:
            temp = np.array(F['{}'.format(tg)])
            temp_index = np.where(temp != -9999)
            if i == 0:
                index_not9999 = temp_index
                i += 1
            else:
                index_not9999 = reduce(np.intersect1d, (index_not9999, temp_index))
        index_not9999 = index_not9999[0:1100]

        test_spectra = np.array(F['spectra'])
        test_spectra = test_spectra[index_not9999]
        test_spectra -= spec_meanstd[0]
        test_spectra /= spec_meanstd[1]

        i = 0
        test_labels = []
        for tg in target:  # load data
            temp = np.array(F['{}'.format(tg)])
            temp = temp[index_not9999]
            if i == 0:
                test_labels = temp[:]
                if len(target) == 1:
                    test_labels = test_labels.reshape((len(test_labels), 1))
                i += 1
            else:
                test_labels = np.column_stack((test_labels, temp[:]))

    prediction = batch_predictions(model, test_spectra, 1000, num_labels, std_labels, mean_labels)

    print('Test set contains ' + str(len(test_spectra)) + ' stars')

    time1 = time.time()
    test_predictions = []
    for j in range(7504):
        with h5py.File(traindata) as F:
            test_spectra = np.array(F['spectra'])
            test_spectra = test_spectra[index_not9999]
            test_spectra -= spec_meanstd[0]
            test_spectra /= spec_meanstd[1]
        temp = test_spectra
        temp[:,j:j+10] = 0
        test_predictions.extend([batch_predictions(model, temp, 1000, num_labels, std_labels, mean_labels) - prediction])
        print(j)
    print("{0:.2f}".format(time.time() - time1) + ' seconds to make ' + str(len(test_spectra)) + ' predictions')

    resid = np.mean(test_predictions, axis=1)
    print(resid.shape)

    # Some plotting variables for asthetics
    plt.rcParams['axes.facecolor'] = 'white'
    sns.set_style("ticks")
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.alpha'] = '0.4'

    for i in range(num_labels):
        plt.figure(figsize=(40, 11), dpi=200)
        plt.axhline(0, ls='--', c='k', lw=2)
        plt.plot(resid[:, i], linewidth=0.7)
        fullname = target_name_conversion(target[i])
        plt.xlabel(fullname, fontsize=25)
        path = os.path.join(fullfolderpath, 'blackbox')
        if not os.path.exists(path):
            os.makedirs(path)
        plt.tick_params(labelsize=20, width=1, length=10)
        plt.tight_layout()
        plt.savefig(path + '/{}.png'.format(target[i]))
        plt.close('all')
        plt.clf()