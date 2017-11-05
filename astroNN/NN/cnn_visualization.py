# ---------------------------------------------------------#
#   astroNN.NN.cnn_visualization: Visualizing CNN model
# ---------------------------------------------------------#

import os
import random
from functools import reduce

import h5py
import matplotlib.colors as colors
import numpy as np
import pylab as plt
from keras import backend as K
from keras.models import load_model

import astroNN.NN.train_tools


def cnn_visualization(folder_name=None, h5name=None, num=None):
    """
    NAME: cnn_visualization
    PURPOSE: To visualize CNN model
    INPUT:
        folder_name = parent folder name
        data =
    OUTPUT: plots
    HISTORY:
        2017-Nov-02 Henry Leung
    """

    # Set number of spectra for CNN visualization
    if num is None:
        num = 20

    random.seed(3)

    data = h5name + '_train.h5'

    currentdir = os.getcwd()
    fullfolderpath = currentdir + '/' + folder_name
    vis_parent_path = os.path.join(fullfolderpath, 'cnn_visual')

    # load model
    modelname = '/model_{}.h5'.format(folder_name[-11:])
    model = load_model(os.path.normpath(fullfolderpath + modelname))

    layer_1 = K.function([model.layers[0].input, K.learning_phase()], [model.layers[1].output])

    layer_2 = K.function([model.layers[0].input, K.learning_phase()], [model.layers[2].output])

    target = np.load(fullfolderpath + '/targetname.npy')
    spec_meanstd = np.load(fullfolderpath + '/spectra_meanstd.npy')

    with h5py.File(data) as F:  # ensure the file will be cleaned up
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

        spectra = np.array(F['spectra'])
        rel_index = np.array(F['index'])
        spectra = spectra[index_not9999]
        spectra -= spec_meanstd[0]
        spectra /= spec_meanstd[1]
        random_number = num
        ran = random.sample(range(0, spectra.shape[0], 1), random_number)
        spectra = spectra[ran]
        rel_index = rel_index[ran]
    num_label = spectra.shape[1]

    for i in range(random_number):
        temp_path = os.path.join(vis_parent_path, str(i))
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        reshaped = spectra[i].reshape((1, num_label, 1))
        layer_1_output = layer_1([reshaped, 0])[0]
        layer_2_output = layer_2([reshaped, 0])[0]
        apogee_id = astroNN.NN.train_tools.apogee_id_fetch(relative_index=rel_index, dr=14)

        plt.figure(figsize=(30, 15), dpi=200)
        plt.rcParams['axes.grid'] = False
        plt.plot(spectra[i] * spec_meanstd[1] + spec_meanstd[0], alpha=0.8, linewidth=0.7, label='APOGEE Spectra')
        plt.xlabel('Pixel', fontsize=25)
        plt.ylabel('Flux ', fontsize=25)
        plt.title(apogee_id[i], fontsize=30)
        plt.xlim((0, num_label))
        plt.ylim((0.5, 1.5))
        plt.tick_params(labelsize=20, width=1, length=10)
        leg = plt.legend(loc='best', fontsize=20)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(4.0)
        plt.tight_layout()
        plt.savefig(temp_path + '/spectra_{}.png'.format(apogee_id[i]))
        plt.close('all')
        plt.clf()

        plt.figure(figsize=(25, 20), dpi=200)
        plt.ylabel('Pixel', fontsize=35)
        plt.xlabel('CNN Filter number', fontsize=35)
        plt.title(apogee_id[i], fontsize=30)
        plt.imshow(layer_1_output[0, :, :], aspect='auto', norm=colors.PowerNorm(gamma=1. / 2.), cmap='gray')
        plt.tick_params(labelsize=25, width=1, length=10)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=25, width=1, length=10)
        plt.tight_layout()
        plt.savefig(temp_path + '/cnn_layer1.png')
        plt.close('all')
        plt.clf()

        plt.figure(figsize=(25, 20), dpi=200)
        plt.ylabel('Pixel', fontsize=35)
        plt.xlabel('CNN Filter number', fontsize=35)
        plt.title(apogee_id[i], fontsize=30)
        plt.imshow(layer_2_output[0, :, :], aspect='auto', norm=colors.PowerNorm(gamma=1. / 2.), cmap='gray')
        plt.tick_params(labelsize=25, width=1, length=10)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=25, width=1, length=10)
        plt.tight_layout()
        plt.savefig(temp_path + '/cnn_layer2.png')
        plt.close('all')
        plt.clf()
