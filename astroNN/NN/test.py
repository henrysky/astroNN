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
import scipy


def apogee_test(model=None, testdata=None, folder_name=None):
    """
    NAME: apogee_test
    PURPOSE: To test the model
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
    for i in range(num_labels):
        fig = plt.figure(figsize=[14, 14], dpi=200)
        bins = [200, 200]  # number of bins
        thresh = 3  # density threshold
        xy_range = [[np.min(test_predictions[:, i])-np.abs(np.min(test_predictions[:, i])*0.2), np.max(test_predictions[:, i])*1.2],
                    [np.min(resid[:, i])-np.abs(np.min(resid[:, i])*0.5), np.max(resid[:, i])*1.5]]  # data range
        hh, locx, locy = scipy.histogram2d(test_predictions[:, i], resid[:, i], range=xy_range, bins=bins)
        posx = np.digitize(test_predictions[:, i], locx)
        posy = np.digitize(resid[:, i], locy)
        # select points within the histogram
        ind = (posx > 0) & (posx <= bins[0]) & (posy > 0) & (posy <= bins[1])
        hhsub = hh[posx[ind] - 1, posy[ind] - 1]  # values of the histogram where the points are
        xdat1 = (test_predictions[:, i])[ind][hhsub < thresh]  # low density points
        ydat1 = (resid[:, i])[ind][hhsub < thresh]
        hh[hh < thresh] = np.nan  # fill the areas with low density by NaNs

        plt.imshow(np.flipud(hh.T), cmap='jet', extent=np.array(xy_range).flatten(), interpolation='none',
                   origin='upper')
        # plt.colorbar()
        plt.plot(xdat1, ydat1, '.', color='darkblue')
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=3)
        plt.figtext(0,0,'$\widetilde{m}$=' + '{0:.3f}'.format(bias[i]) + ' $s$=' + '{0:.3f}'.format(scatter[i]/std_labels[i]),
                    size=10, bbox=bbox_props)
        plt.savefig(folder_name + '{}_test.png'.format(target[i]))

    return None
