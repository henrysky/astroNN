# ---------------------------------------------------------#
#   astroNN.NN.cnn_visualization: Visualizing CNN model
# ---------------------------------------------------------#

import random
import pylab as plt
from keras import backend as K
from keras.models import load_model
import h5py
import numpy as np
from functools import reduce
import os


def cnn_visualization(model=None, data=None):
    """
    NAME: cnn_visualization
    PURPOSE: To visualize CNN model
    INPUT: model in absolute path
    OUTPUT: plots
    HISTORY:
        2017-Nov-02 Henry Leung
    """

    currentdir = os.getcwd()
    vis_parent_path = os.path.join(currentdir, 'cnn_visual')

    # load model
    model = load_model(model)

    layer_1 = K.function([model.layers[0].input, K.learning_phase()], [model.layers[1].output])

    layer_2 = K.function([model.layers[0].input, K.learning_phase()], [model.layers[2].output])

    target = np.load('targetname.npy')
    spec_meanstd = np.load('spectra_meanstd.npy')

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
        spectra = spectra[index_not9999]
        spectra -= spec_meanstd[0]
        spectra /= spec_meanstd[1]
        random_number = 20
        ran = random.sample(range(0, spectra.shape[0], 1), random_number)
        spectra = spectra[ran]
    num_label = spectra.shape[1]

    for i in range(random_number):
        temp_path = os.path.join(vis_parent_path, str(i))
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        reshaped = spectra[i].reshape((1,num_label,1))
        layer_1_output = layer_1([reshaped, 0])[0]
        layer_2_output = layer_2([reshaped, 0])[0]

        plt.figure(figsize=(30, 11), dpi=200)
        plt.plot(spectra[i] * spec_meanstd[1] + spec_meanstd[0], alpha=0.8, linewidth=0.7, label='APOGEE Spectra')
        plt.xlabel('Pixel', fontsize=25)
        plt.ylabel('Flux ', fontsize=25)
        plt.xlim((0,num_label))
        plt.ylim((0.5,1.5))
        plt.tick_params(labelsize=20, width=1, length=10)
        plt.tight_layout()
        leg = plt.legend(loc='best', fontsize=20)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(4.0)
        plt.savefig(temp_path + '/spectra_{}.png'.format(i))
        plt.close('all')
        plt.clf()

        plt.figure(figsize=(20, 20), dpi=200)
        plt.ylabel('Pixel', fontsize=30)
        plt.xlabel('CNN Filter number', fontsize=30)
        plt.imshow(layer_1_output[0,:,:], aspect='auto')
        plt.tick_params(labelsize=20, width=1, length=10)
        plt.savefig(temp_path + '/cnn_layer1.png')
        plt.close('all')
        plt.clf()

        plt.figure(figsize=(20, 20), dpi=200)
        plt.ylabel('Pixel', fontsize=30)
        plt.xlabel('CNN Filter number', fontsize=30)
        plt.imshow(layer_2_output[0,:,:], aspect='auto')
        plt.tick_params(labelsize=20, width=1, length=10)
        plt.savefig(temp_path + '/cnn_layer2.png')
        plt.close('all')
        plt.clf()



