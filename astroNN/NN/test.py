# ---------------------------------------------------------#
#   astroNN.NN.test: test models
# ---------------------------------------------------------#

import os
import time
from functools import reduce

import h5py
import numpy as np
import pylab as plt
import seaborn as sns
import tensorflow as tf
from astropy.stats import mad_std
from keras.backend.tensorflow_backend import set_session

from keras.models import load_model

import astroNN.apogee.cannon
from astroNN.shared.nn_tools import h5name_check, foldername_modelname, batch_predictions, batch_dropout_predictions\
    , target_name_conversion, gpu_memory_manage
import astroNN.NN.jacobian
# from astroNN.NN.jacobian import prop_err, keras_to_tf, cal_jacobian
# from astroNN.NN.train_tools import apogee_id_fetch


def target_to_aspcap_conversion(targetname):
    if targetname == 'alpha':
        fullname = targetname + '_M'
    elif len(targetname) < 3:
        fullname = targetname + '_H'
    else:
        fullname = targetname
    return fullname


def apogee_model_eval(h5name=None, folder_name=None, check_cannon=None, test_noisy=None):
    """
    NAME: apogee_model_eval
    PURPOSE: To test the model and generate plots
    INPUT:
        h5name = Name of the h5 data set
        folder_name = the folder name contains the model
        check_cannon = check cannon result or not
        test_noist = whether test noisy training data or not (both adding noise and transolational shift)
    OUTPUT: plots
    HISTORY:
        2017-Oct-14 Henry Leung
    """

    # prevent Tensorflow taking up all the GPU memory
    gpu_memory_manage()

    h5name_check(h5name)
    model_tf = astroNN.NN.jacobian.keras_to_tf(folder_name=folder_name)


    if test_noisy is None:
        test_noisy = False

    h5test = h5name + '_test.h5'
    traindata = h5name + '_train.h5'

    currentdir = os.getcwd()
    fullfolderpath = currentdir + '/' + folder_name
    mean_and_std = np.load(fullfolderpath + '/meanstd.npy')
    spec_meanstd = np.load(fullfolderpath + '/spectra_meanstd.npy')
    target = np.load(fullfolderpath + '/targetname.npy')
    modelname = foldername_modelname(folder_name=folder_name)
    model = load_model(os.path.normpath(fullfolderpath + modelname))

    mean_labels = mean_and_std[0]
    std_labels = mean_and_std[1]
    num_labels = mean_and_std.shape[1]

    # ensure the file will be cleaned up
    with h5py.File(h5test) as F:
        index_not9999 = []
        for counter, tg in enumerate(target):
            temp = np.array(F['{}'.format(tg)])
            temp_index = np.where(temp != -9999)
            if counter == 0:
                index_not9999 = temp_index
            else:
                index_not9999 = reduce(np.intersect1d, (index_not9999, temp_index))

        index_not9999 = index_not9999[0:500]

        test_spectra = np.array(F['spectra'])
        test_spectra_err = np.array(F['spectra_err'])
        test_spectra = test_spectra[index_not9999]
        test_spectra_err = test_spectra_err[index_not9999]
        test_spectra -= spec_meanstd[0]
        test_spectra /= spec_meanstd[1]

        test_labels = []
        for counter, tg in enumerate(target):  # load data
            temp = np.array(F['{}'.format(tg)])
            temp = temp[index_not9999]
            if counter == 0:
                test_labels = temp[:]
                if len(target) == 1:
                    test_labels = test_labels.reshape((len(test_labels), 1))
            else:
                test_labels = np.column_stack((test_labels, temp[:]))
        apogee_index = np.array(F['index'])[index_not9999]

    print('Test set contains ' + str(len(test_spectra)) + ' stars')

    time1 = time.time()
    test_predictions, model_uncertainty = batch_dropout_predictions(model, test_spectra, 500, num_labels, std_labels,
                                                                    mean_labels)
    properr = astroNN.NN.jacobian.prop_err(model_tf, test_spectra.reshape((len(test_spectra), 7514, 1)), std_labels,
                                           mean_labels, test_spectra_err)
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
        plt.errorbar(test_labels[:, i], resid[:, i], xerr=properr[:, i], yerr=model_uncertainty[:, i], markersize=2,
                     fmt='o', ecolor='g', capthick=2, elinewidth=1)

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
            index_not9999 = []
            for counter, tg in enumerate(target):
                temp = np.array(F['{}'.format(tg)])
                temp_index = np.where(temp != -9999)
                if counter == 0:
                    index_not9999 = temp_index
                else:
                    index_not9999 = reduce(np.intersect1d, (index_not9999, temp_index))

            train_spectra = np.array(F['spectra'])
            train_spectra = train_spectra[index_not9999]
            sigma = 0.08 ** 2
            train_spectra_noisy = train_spectra + np.random.poisson(sigma, train_spectra.shape)
            train_spectra -= spec_meanstd[0]
            train_spectra /= spec_meanstd[1]
            train_spectra_noisy -= spec_meanstd[0]
            train_spectra_noisy /= spec_meanstd[1]
            random_num_color = np.array([])
            for index in range(train_spectra_noisy.shape[0]):
                # make sure no 0 pixel shift
                while True:
                    random_temp = np.random.randint(-7, 7)
                    if random_temp != 0:
                        break
                random_num_color = np.append(random_num_color, random_temp)
                train_spectra_noisy[index] = np.roll(train_spectra_noisy[index], random_temp)
            for counter, tg in enumerate(target):  # load data
                temp = np.array(F['{}'.format(tg)])
                temp = temp[index_not9999]
                if counter == 0:
                    train_labels = temp[:]
                    if len(target) == 1:
                        train_labels = train_labels.reshape((len(train_labels), 1))
                else:
                    train_labels = np.column_stack((train_labels, temp[:]))

        if test_noisy is True:
            train_noisy_predictions = batch_predictions(model, train_spectra_noisy, 500, num_labels, std_labels,
                                                        mean_labels)
            train_predictions = batch_predictions(model, train_spectra, 500, num_labels, std_labels,
                                                  mean_labels)
            resid_noisy = train_noisy_predictions - train_labels
            resid_train = train_predictions - train_labels
            bias_train = np.median(resid_train, axis=0)
            scatter_train = mad_std(resid_train, axis=0)
            bias_noisy = np.median(resid_noisy, axis=0)
            scatter_noisy = mad_std(resid_noisy, axis=0)

            # Some plotting variables for aesthetics
            plt.rcParams['axes.facecolor'] = 'white'
            sns.set_style("ticks")
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.color'] = 'gray'
            plt.rcParams['grid.alpha'] = '0.4'

            x_lab = 'ASPCAP'
            y_lab = 'astroNN'
            trainplot_noisy_fullpath = os.path.join(fullfolderpath, 'Noisy_TrainData_Plots/')
            trainplot_noisy_2_fullpath = os.path.join(fullfolderpath, 'Noisy_TrainData_Plots_02/')
            trainplot_fullpath = os.path.join(fullfolderpath, 'TrainData_Plots/')

            # check folder existence
            if not os.path.exists(trainplot_fullpath):
                os.makedirs(trainplot_fullpath)
            if not os.path.exists(trainplot_noisy_fullpath):
                os.makedirs(trainplot_noisy_fullpath)
            if not os.path.exists(trainplot_noisy_2_fullpath):
                os.makedirs(trainplot_noisy_2_fullpath)

            for i in range(num_labels):
                plt.figure(figsize=(15, 11), dpi=200)
                plt.axhline(0, ls='--', c='k', lw=2)
                plt.scatter(train_labels[:, i], resid_train[:, i], s=3)
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
                            '$\widetilde{m}$=' + '{0:.3f}'.format(
                                bias_train[i]) + ' $\widetilde{s}$=' + '{0:.3f}'.format(
                                scatter_train[i] / std_labels[i]) + ' s=' + '{0:.3f}'.format(scatter_train[i]), size=25,
                            bbox=bbox_props)
                plt.tight_layout()
                plt.savefig(trainplot_fullpath + '{}_train_data.png'.format(target[i]))
                plt.close('all')
                plt.clf()

                plt.figure(figsize=(15, 11), dpi=200)
                plt.axhline(0, ls='--', c='k', lw=2)
                plt.scatter(train_labels[:, i], resid_noisy[:, i], c=random_num_color, s=3, cmap='gray')
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
                plt.figtext(0.5, 0.85,
                            '$\widetilde{m}$=' + '{0:.3f}'.format(
                                bias_noisy[i]) + ' $\widetilde{s}$=' + '{0:.3f}'.format(
                                scatter_noisy[i] / std_labels[i]) + ' s=' + '{0:.3f}'.format(scatter_noisy[i]), size=25,
                            bbox=bbox_props)
                cbar = plt.colorbar()
                cbar.ax.tick_params(labelsize=25, width=1, length=10)
                plt.tight_layout()
                plt.savefig(trainplot_noisy_fullpath + '{}_noisytrain_data.png'.format(target[i]))
                plt.close('all')
                plt.clf()
    if check_cannon is True:
        astroNN.apogee.cannon.cannon_plot(apogee_index, std_labels, target, folder_name=folder_name,
                                               aspcap_answer=test_labels)

    return None


def gaia_model_eval(h5name=None, folder_name=None):
    """
    NAME: apogee_model_eval
    PURPOSE: To test the model and generate plots
    INPUT:
        h5name = Name of the h5 data set
        folder_name = the folder name contains the model
    OUTPUT: plots
    HISTORY:
        2017-Oct-14 Henry Leung
    """

    # prevent Tensorflow taking up all the GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    h5name_check(h5name)

    h5test = h5name + '_test.h5'
    traindata = h5name + '_train.h5'

    currentdir = os.getcwd()
    fullfolderpath = currentdir + '/' + folder_name
    print(fullfolderpath)
    mean_and_std = np.load(fullfolderpath + '/meanstd.npy')
    spec_meanstd = np.load(fullfolderpath + '/spectra_meanstd.npy')
    modelname = foldername_modelname(folder_name=folder_name)
    model = load_model(os.path.normpath(fullfolderpath + modelname))

    mean_labels = mean_and_std[0]
    std_labels = mean_and_std[1]

    # ensure the file will be cleaned up
    with h5py.File(h5test) as F:
        test_spectra = np.array(F['spectra'])
        test_spectra -= spec_meanstd[0]
        test_spectra /= spec_meanstd[1]
        absmag = np.array(F['absmag'])

    print('Test set contains ' + str(len(test_spectra)) + ' stars')

    time1 = time.time()
    test_predictions = batch_predictions(model, test_spectra, 500, 1, std_labels, mean_labels)
    print("{0:.2f}".format(time.time() - time1) + ' seconds to make ' + str(len(test_spectra)) + ' predictions')

    resid = test_predictions.flatten() - absmag
    bias = np.median(resid)
    scatter = mad_std(resid)

    # Some plotting variables for asthetics
    plt.rcParams['axes.facecolor'] = 'white'
    sns.set_style("ticks")
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.alpha'] = '0.4'

    x_lab = 'Gaia'
    y_lab = 'astroNN'
    plt.figure(figsize=(15, 11), dpi=200)
    plt.axhline(0, ls='--', c='k', lw=2)
    plt.scatter(absmag, resid, s=3)
    plt.xlabel('Gaia Abs Mag', fontsize=25)
    plt.ylabel('$\Delta$ Abs Mag' + '\n(' + y_lab + ' - ' + x_lab + ')', fontsize=25)
    plt.tick_params(labelsize=20, width=1, length=10)
    plt.xlim([np.min(absmag), np.max(absmag)])
    ranges = (np.max(absmag) - np.min(absmag)) / 2
    plt.ylim([-ranges, ranges])
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=2)
    plt.figtext(0.6, 0.75, '$\widetilde{m}$=' + '{0:.3f}'.format(float(bias)) + ' $\widetilde{s}$=' + '{0:.3f}'.format(
        float(scatter / std_labels)) + ' s=' + '{0:.3f}'.format(float(scatter)), size=25, bbox=bbox_props)
    plt.tight_layout()
    plt.savefig(fullfolderpath + '/absmag_test.png')
    plt.close('all')
    plt.clf()

    if traindata is not None:
        with h5py.File(traindata) as F:
            train_spectra = np.array(F['spectra'])
            train_spectra -= spec_meanstd[0]
            train_spectra /= spec_meanstd[1]
            absmag = np.array(F['absmag'])

        time1 = time.time()
        test_predictions = batch_predictions(model, train_spectra, 500, 1, std_labels, mean_labels)
        print("{0:.2f}".format(time.time() - time1) + ' seconds to make ' + str(len(train_spectra)) + ' predictions')

        resid = test_predictions.flatten() - absmag
        bias = np.median(resid)
        scatter = mad_std(resid)

        # Some plotting variables for asthetics
        plt.rcParams['axes.facecolor'] = 'white'
        sns.set_style("ticks")
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.color'] = 'gray'
        plt.rcParams['grid.alpha'] = '0.4'

        x_lab = 'Gaia'
        y_lab = 'astroNN'
        plt.figure(figsize=(15, 11), dpi=200)
        plt.axhline(0, ls='--', c='k', lw=2)
        plt.scatter(absmag, resid, s=3)
        plt.xlabel('Gaia Abs Mag', fontsize=25)
        plt.ylabel('$\Delta$ Abs Mag' + '\n(' + y_lab + ' - ' + x_lab + ')', fontsize=25)
        plt.tick_params(labelsize=20, width=1, length=10)
        plt.xlim([np.min(absmag), np.max(absmag)])
        ranges = (np.max(absmag) - np.min(absmag)) / 2
        plt.ylim([-ranges, ranges])
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=2)
        plt.figtext(0.6, 0.75,
                    '$\widetilde{m}$=' + '{0:.3f}'.format(float(bias)) + ' $\widetilde{s}$=' + '{0:.3f}'.format(
                        float(scatter / std_labels)) + ' s=' + '{0:.3f}'.format(float(scatter)), size=25,
                    bbox=bbox_props)
        plt.tight_layout()
        plt.savefig(fullfolderpath + '/absmag_train.png')
        plt.close('all')
        plt.clf()

    return None
