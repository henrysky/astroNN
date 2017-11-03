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
from functools import reduce
import seaborn as sns
import astroNN.apogeetools.cannon
import os
from astropy.stats import mad_std


def batch_predictions(model, spectra, batch_size, num_labels, std_labels, mean_labels):
    predictions = np.zeros((len(spectra), num_labels))
    i = 0
    for i in range(len(spectra) // batch_size):
        inputs = spectra[i * batch_size:(i + 1) * batch_size].reshape((batch_size, spectra.shape[1], 1))
        predictions[i * batch_size:(i + 1) * batch_size] = denormalize(model.predict(inputs), std_labels, mean_labels)
    inputs = spectra[(i + 1) * batch_size:].reshape((spectra[(i + 1) * batch_size:].shape[0], spectra.shape[1], 1))
    predictions[(i + 1) * batch_size:] = denormalize(model.predict(inputs), std_labels, mean_labels)
    return predictions


def denormalize(lb_norm, std_labels, mean_labels):
    return ((lb_norm * std_labels) + mean_labels)


def target_name_conversion(targetname):
    if len(targetname) < 3:
        fullname = '[{}/H]'.format(targetname)
    elif targetname == 'teff':
        fullname = '$T_{\mathrm{eff}}$'
    elif targetname == 'alpha':
        fullname = '[Alpha/M]'
    elif targetname == 'logg':
        fullname = '[Log(g)]'
    else:
        fullname = targetname
    return fullname


def target_to_aspcap_conversion(targetname):
    if targetname == 'alpha':
        fullname = targetname + '_M'
    elif len(targetname) < 3:
        fullname = targetname + '_H'
    else:
        fullname = targetname
    return fullname


def apogee_model_eval(h5name=None, folder_name=None, check_cannon=None):
    """
    NAME: apogee_model_eval
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

    if h5name is None or folder_name is None:
        raise ValueError('Please specify testdata or folder_name')

    h5test = h5name + '_test.h5'
    traindata = h5name + '_test.h5'

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
    with h5py.File(h5test) as F:
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

        test_spectra = np.array(F['spectra'])
        test_spectra = test_spectra[index_not9999]
        test_spectra -= spec_meanstd[0]
        test_spectra /= spec_meanstd[1]
        i = 0
        test_labels = np.array((test_spectra.shape[1]))
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
        apogee_index = np.array(F['index'])[index_not9999]

    print('Test set contains ' + str(len(test_spectra)) + ' stars')

    time1 = time.time()
    test_predictions = batch_predictions(model, test_spectra, 500, num_labels, std_labels, mean_labels)
    print("{0:.2f}".format(time.time() - time1) + ' seconds to make ' + str(len(test_spectra)) + ' predictions')

    resid = test_predictions - test_labels
    bias = np.median(resid, axis=0)
    scatter = mad_std(resid, axis=0)

    # Some plotting variables for asthetics
    plt.rcParams['axes.facecolor'] = 'white'
    sns.set_style("ticks")
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.alpha'] = '0.4'

    x_lab = 'ASPCAP'
    y_lab = 'astroNN'
    for i in range(num_labels):
        plt.figure(figsize=(15, 11), dpi=200)
        plt.axhline(0, ls='--', c='k', lw=2)
        plt.scatter(test_labels[:, i], resid[:, i], s=3)
        fullname = target_name_conversion(target[i])
        plt.xlabel('ASPCAP ' + fullname, fontsize=25)
        plt.ylabel('$\Delta$ ' + fullname + '\n(' + y_lab + ' - ' + x_lab + ')', fontsize=25)
        plt.tick_params(labelsize=20, width=1, length=10)
        if num_labels == 1:
            plt.xlim([np.min(test_labels[:]), np.max(test_labels[:])])
        else:
            plt.xlim([np.min(test_labels[:, i]), np.max(test_labels[:, i])])
        ranges = (np.max(test_labels[:, i]) - np.min(test_labels[:, i])) / 2
        plt.ylim([-ranges, ranges])
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=2)
        plt.figtext(0.6, 0.75, '$\widetilde{m}$=' + '{0:.3f}'.format(bias[i]) + ' $\widetilde{s}$=' + '{0:.3f}'.format(
            scatter[i] / std_labels[i]) + ' s=' + '{0:.3f}'.format(scatter[i]), size=25, bbox=bbox_props)
        plt.tight_layout()
        plt.savefig(fullfolderpath + '/{}_test.png'.format(target[i]))
        plt.close('all')
        plt.clf()

    if traindata is not None:
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

            train_spectra = np.array(F['spectra'])
            train_spectra = train_spectra[index_not9999]
            sigma = 0.04
            train_spectra_noisy = train_spectra + np.random.normal(0, sigma, train_spectra.shape)
            train_spectra -= spec_meanstd[0]
            train_spectra /= spec_meanstd[1]
            train_spectra_noisy -= spec_meanstd[0]
            train_spectra_noisy /= spec_meanstd[1]
            i = 0
            train_labels = np.array((train_spectra.shape[1]))
            for tg in target:  # load data
                temp = np.array(F['{}'.format(tg)])
                temp = temp[index_not9999]
                if i == 0:
                    train_labels = temp[:]
                    if len(target) == 1:
                        train_labels = train_labels.reshape((len(train_labels), 1))
                    i += 1
                else:
                    train_labels = np.column_stack((train_labels, temp[:]))

        test_predictions = batch_predictions(model, train_spectra, 500, num_labels, std_labels, mean_labels)
        resid = test_predictions - train_labels
        bias = np.median(resid, axis=0)
        scatter = np.std(resid, axis=0)

        test_noisy_predictions = batch_predictions(model, train_spectra_noisy, 500, num_labels, std_labels, mean_labels)
        resid_noisy = test_noisy_predictions - train_labels
        bias_noisy = np.median(resid_noisy, axis=0)
        scatter_noisy = np.std(resid_noisy, axis=0)

        # Some plotting variables for aesthetics
        plt.rcParams['axes.facecolor'] = 'white'
        sns.set_style("ticks")
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.color'] = 'gray'
        plt.rcParams['grid.alpha'] = '0.4'

        x_lab = 'ASPCAP'
        y_lab = 'astroNN'
        trainplot_noisy_fullpath = os.path.join(fullfolderpath, 'Noisy_TrainData_Plots/')
        print(trainplot_noisy_fullpath)
        trainplot_fullpath = os.path.join(fullfolderpath, 'TrainData_Plots/')

        # check folder existence
        if not os.path.exists(trainplot_fullpath):
            os.makedirs(trainplot_fullpath)
        if not os.path.exists(trainplot_noisy_fullpath):
            os.makedirs(trainplot_noisy_fullpath)

        for i in range(num_labels):
            plt.figure(figsize=(15, 11), dpi=200)
            plt.axhline(0, ls='--', c='k', lw=2)
            plt.scatter(train_labels[:, i], resid[:, i], s=3)
            fullname = target_name_conversion(target[i])
            plt.xlabel('ASPCAP ' + fullname, fontsize=25)
            plt.ylabel('$\Delta$ ' + fullname + '\n(' + y_lab + ' - ' + x_lab + ')', fontsize=25)
            plt.tick_params(labelsize=20, width=1, length=10)
            if num_labels == 1:
                plt.xlim([np.min(train_labels[:]), np.max(train_labels[:])])
            else:
                plt.xlim([np.min(train_labels[:, i]), np.max(train_labels[:, i])])
            ranges = (np.max(train_labels[:, i]) - np.min(train_labels[:, i])) / 2
            plt.ylim([-ranges, ranges])
            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=2)
            plt.figtext(0.6, 0.75,
                        '$\widetilde{m}$=' + '{0:.3f}'.format(bias[i]) + ' $\widetilde{s}$=' + '{0:.3f}'.format(
                            scatter[i] / std_labels[i]) + ' s=' + '{0:.3f}'.format(scatter[i]), size=25,
                        bbox=bbox_props)
            plt.tight_layout()
            plt.savefig(trainplot_fullpath + '{}_train_data.png'.format(target[i]))
            plt.close('all')
            plt.clf()

            plt.figure(figsize=(15, 11), dpi=200)
            plt.axhline(0, ls='--', c='k', lw=2)
            plt.scatter(train_labels[:, i], resid_noisy[:, i], s=3)
            fullname = target_name_conversion(target[i])
            plt.xlabel('ASPCAP ' + fullname, fontsize=25)
            plt.ylabel('$\Delta$ ' + fullname + '\n(' + y_lab + ' - ' + x_lab + ')', fontsize=25)
            plt.tick_params(labelsize=20, width=1, length=10)
            if num_labels == 1:
                plt.xlim([np.min(train_labels[:]), np.max(train_labels[:])])
            else:
                plt.xlim([np.min(train_labels[:, i]), np.max(train_labels[:, i])])
            ranges = (np.max(train_labels[:, i]) - np.min(train_labels[:, i])) / 2
            plt.ylim([-ranges, ranges])
            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=2)
            plt.figtext(0.6, 0.75,
                        '$\widetilde{m}$=' + '{0:.3f}'.format(bias_noisy[i]) + ' $\widetilde{s}$=' + '{0:.3f}'.format(
                            scatter_noisy[i] / std_labels[i]) + ' s=' + '{0:.3f}'.format(scatter_noisy[i]), size=25,
                        bbox=bbox_props)
            plt.tight_layout()
            plt.savefig(trainplot_noisy_fullpath + '{}_noisytrain_data.png'.format(target[i]))
            plt.close('all')
            plt.clf()

    if check_cannon is True:
        astroNN.apogeetools.cannon.cannon_plot(apogee_index, num_labels, std_labels, target, folder_name=folder_name,
                                               aspcap_answer=test_labels)

    return None
