# ---------------------------------------------------------#
#   astroNN.NN.generative_test: test generative models
# ---------------------------------------------------------#

import random
import time

import h5py
import numpy as np
import pylab as plt
import seaborn as sns
import tensorflow as tf
from astropy.io import fits
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model

import astroNN.NN.train_tools
import astroNN.datasets.h5_compiler
from astroNN.shared.nn_tools import gpu_memory_manage


def apogee_generative_test(model=None, testdata=None, folder_name=None, std=None):
    """
    NAME: apogee_generative_test
    PURPOSE: To test the model and generate plots
    INPUT:
    OUTPUT: target and normalized data
    HISTORY:
        2017-Oct-14 Henry Leung
    """

    # prevent Tensorflow taking up all the GPU memory
    gpu_memory_manage()

    if testdata is None or folder_name is None:
        raise ValueError('Please specify testdata or folder_name')

    with h5py.File(testdata) as F:
        random_number = 20
        test_spectra = np.array(F['spectra'])
        ran = random.sample(range(0, test_spectra.shape[0], 1), random_number)
        bestfit_spectra = np.array(F['spectrabestfit'])
        rel_index = np.array((F['index'])[ran])
        test_spectra = test_spectra[ran]
        bestfit_spectra = bestfit_spectra[ran]
    num_labels = test_spectra.shape[1]
    print('Test set contains ' + str(len(test_spectra)) + ' stars')
    model = load_model(model)

    # Some plotting variables for asthetics
    plt.rcParams['axes.facecolor'] = 'white'
    sns.set_style("ticks")
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.alpha'] = '0.4'

    for i in range(random_number):
        apogee_id = astroNN.NN.train_tools.apogee_id_fetch(relative_index=rel_index, dr=14)
        test_predictions = predictions(model, test_spectra[i], std)
        test_predictions = test_predictions.reshape(num_labels)
        plt.figure(figsize=(30, 11), dpi=200)
        plt.plot(bestfit_spectra[i], linewidth=0.7, label='ASCPCAP Bestfit')
        plt.plot(test_spectra[i] * std[0] + 1, alpha=0.5, linewidth=0.7, label='APOGEE combined Spectra')
        plt.plot(test_predictions * std[0] + 1, alpha=0.5, linewidth=0.7, label='astroNN generative model')
        plt.xlabel('Pixel', fontsize=25)
        plt.ylabel('Flux ', fontsize=25)
        plt.title(apogee_id[i], fontsize=30)
        plt.xlim((0, num_labels))
        plt.ylim((0.5, 1.5))
        plt.tick_params(labelsize=20, width=1, length=10)
        plt.tight_layout()
        leg = plt.legend(loc='best', fontsize=20)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(4.0)
        plt.savefig(folder_name + '{}_test.png'.format(apogee_id[i]))
        plt.close('all')
        plt.clf()

    return None


def predictions(model, spectra, std):
    inputs = spectra.reshape((1, 7514, 1))
    inputs -= 1
    inputs /= std[0]
    predictions = model.predict(inputs)
    predictions *= std[1]
    predictions += 1
    return predictions


def apogee_generative_fitstest(model=None, fitsdata=None, std=None):
    """
    NAME: apogee_generative_test
    PURPOSE: To test the model and generate plots
    INPUT:
    OUTPUT: target and normalized data
    HISTORY:
        2017-Oct-14 Henry Leung
    """

    # prevent Tensorflow taking up all the GPU memory
    gpu_memory_manage()

    if fitsdata is None:
        raise ValueError('Please specify testdata')

    with fits.open(fitsdata) as F:
        _spec = np.array(F[1].data)  # Pseudo-comtinumm normalized flux
        _spec_bestfit = np.array(F[3].data)  # Best fit spectrum for training generative model
        _spec = astroNN.datasets.h5_compiler.gap_delete(_spec, dr=14)  # Delete the gap between sensors
        _spec_bestfit = astroNN.datasets.h5_compiler.gap_delete(_spec_bestfit, dr=14)  # Delete the gap between sensors
        print(_spec)
        _spec = _spec.reshape((7514, 1))
    num_labels = _spec.shape[0]
    print('Test set contains ' + str(len(_spec)) + ' stars')
    model = load_model(model)

    time1 = time.time()
    test_predictions = predictions(model, _spec, std)
    print("{0:.2f}".format(time.time() - time1) + ' seconds to make ' + str(len(_spec)) + ' predictions')

    # Some plotting variables for asthetics
    plt.rcParams['axes.facecolor'] = 'white'
    sns.set_style("ticks")
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.alpha'] = '0.4'

    plt.figure(figsize=(30, 11), dpi=200)
    plt.plot(_spec_bestfit, linewidth=0.7, label='ASCPCAP Bestfit', )
    plt.plot(_spec, alpha=0.5, linewidth=0.7, label='APOGEE combined Spectra')
    test_predictions = test_predictions.reshape(7514)
    plt.plot(test_predictions, alpha=0.5, linewidth=0.7, label='astroNN generative model')
    plt.xlabel('Pixel', fontsize=25)
    plt.ylabel('Flux ', fontsize=25)
    plt.xlim((0, 7514))
    plt.ylim((0.5, 1.5))
    plt.tick_params(labelsize=20, width=1, length=10)
    plt.tight_layout()
    leg = plt.legend(loc='best', fontsize=20)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(4.0)
    plt.savefig('{}_test.png'.format(1))
    plt.close('all')
    plt.clf()

    return None
