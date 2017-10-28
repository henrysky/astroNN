# ---------------------------------------------------------#
#   astroNN.NN.generative_test: test generative models
# ---------------------------------------------------------#

import tensorflow as tf
import h5py
import pylab as plt
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
import time
import seaborn as sns
from astropy.io import fits
import astroNN.datasets.h5_compiler


def batch_predictions(model, spectra, batch_size, num_labels):
    predictions = np.zeros((len(spectra), num_labels))
    i = 0
    for i in range(len(spectra) // batch_size):
        inputs = spectra[i * batch_size:(i + 1) * batch_size].reshape((batch_size, spectra.shape[1], 1))
        predictions[i * batch_size:(i + 1) * batch_size] = model.predict(inputs)
    inputs = spectra[(i + 1) * batch_size:].reshape((spectra[(i + 1) * batch_size:].shape[0], spectra.shape[1], 1))
    predictions[(i + 1) * batch_size:] = model.predict(inputs)
    return predictions


def apogee_generative_test(model=None, testdata=None, folder_name=None):
    """
    NAME: apogee_generative_test
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

    if testdata is None or folder_name is None:
        raise ValueError('Please specify testdata or folder_name')

    with h5py.File(testdata) as F:
        test_spectra = np.array(F['spectra'])
        bestfit_spectra = np.array(F['spectrabestfit'])
    num_labels = test_spectra.shape[1]
    print('Test set contains ' + str(len(test_spectra)) + ' stars')
    model = load_model(model)

    time1 = time.time()
    test_predictions = batch_predictions(model, test_spectra, 500, num_labels)
    print("{0:.2f}".format(time.time() - time1) + ' seconds to make ' + str(len(test_spectra)) + ' predictions')

    # Some plotting variables for asthetics
    plt.rcParams['axes.facecolor'] = 'white'
    sns.set_style("ticks")
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.alpha'] = '0.4'

    for i in range(test_spectra.shape[0]):
        plt.figure(figsize=(15, 11), dpi=200)
        plt.axhline(0, ls='--', c='k', lw=2)
        plt.plot(bestfit_spectra, label='ASCPCAP Bestfit')
        plt.plot(test_spectra, label='APOGEE combined Spectra')
        plt.plot(test_predictions[:, i], label='astroNN generative model')
        plt.xlabel('Pixel', fontsize=25)
        plt.ylabel('Flux ', fontsize=25)
        plt.tick_params(labelsize=20, width=1, length=10)
        plt.tight_layout()
        plt.legend(loc='best')
        plt.savefig(folder_name + '{}_test.png'.format(i))
        plt.close('all')
        plt.clf()

    return None

def predictions(model, spectra):
    inputs = spectra.reshape((1,7514,1))
    predictions = model.predict(inputs)
    return predictions

def apogee_generative_fitstest(model=None, fitsdata=None):
    """
    NAME: apogee_generative_test
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
    test_predictions = predictions(model, _spec)
    print("{0:.2f}".format(time.time() - time1) + ' seconds to make ' + str(len(_spec)) + ' predictions')

    # Some plotting variables for asthetics
    plt.rcParams['axes.facecolor'] = 'white'
    sns.set_style("ticks")
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.alpha'] = '0.4'

    plt.figure(figsize=(15, 11), dpi=200)
    plt.plot(_spec_bestfit, label='ASCPCAP Bestfit',)
    plt.plot(_spec, alpha=0.5,label='APOGEE combined Spectra')
    test_predictions = test_predictions.reshape(7514)
    plt.plot(test_predictions, alpha=0.5, label='astroNN generative model')
    plt.xlabel('Pixel', fontsize=25)
    plt.ylabel('Flux ', fontsize=25)
    plt.tick_params(labelsize=20, width=1, length=10)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig('{}_test.png'.format(1))
    plt.close('all')
    plt.clf()

    return None
