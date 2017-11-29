# ---------------------------------------------------------#
#   astroNN.NN.jacobian: calculate NN jacobian
# ---------------------------------------------------------#

import os
from functools import reduce
from urllib.request import urlopen

import h5py
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import tensorflow as tf
from keras.backend.tensorflow_backend import get_session, set_learning_phase, clear_session
from keras.models import load_model
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util

# from astroNN.NN.test import target_name_conversion
from astroNN.apogee.apogee_chips import wavelegnth_solution, chips_split
from astroNN.shared.nn_tools import h5name_check, foldername_modelname, denormalize, target_name_conversion, \
    aspcap_windows_url_correction, gpu_memory_manage


def keras_to_tf(folder_name=None):
    """
    NAME: keras_to_tf
    PURPOSE: Convert keras model to tf graph
    INPUT:
        model = Name of the h5 data set
        folder_name = the folder name contains the model
    OUTPUT: plots
    HISTORY:
        2017-Nov-20 Henry Leung

    Copyright (c) 2017, by the Authors: Amir H. Abdi
    This software is freely available under the MIT Public License.
    Please see the License file in the root for details.
    https://github.com/amir-abdi/keras_to_tensorflow/
    """

    currentdir = os.getcwd()
    fullfolderpath = currentdir + '/' + folder_name
    set_learning_phase(0)

    num_output = 1
    prefix_output_node_names_of_final_network = 'output_node'
    output_graph_name = 'keras_to_tf.pb'

    modelname = foldername_modelname(folder_name=folder_name)
    net_model = load_model(os.path.normpath(fullfolderpath + modelname))

    pred = [None] * num_output
    pred_node_names = [None] * num_output
    for i in range(num_output):
        pred_node_names[i] = prefix_output_node_names_of_final_network + str(i)
        pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
    print('output nodes names are: ', pred_node_names)

    sess = get_session()

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, fullfolderpath, output_graph_name, as_text=False)
    print('saved the constant graph (ready for inference) at: ', os.path.join(fullfolderpath, output_graph_name))

    clear_session()

    return fullfolderpath + '/' + output_graph_name


def load_graph(frozen_graph_filename):
    """
    NAME: load_graph
    PURPOSE: load tf graph
    INPUT:
    OUTPUT:
    HISTORY:
        2017-Nov-20 Henry Leung

    Copyright (c) 2017, by the Authors: Amir H. Abdi
    This software is freely available under the MIT Public License.
    Please see the License file in the root for details.
    https://github.com/amir-abdi/keras_to_tensorflow/
    """

    with tf.gfile.GFile(frozen_graph_filename, "rb") as tfg:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(tfg.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )

    input_name = graph.get_operations()[0].name + ':0'
    output_name = graph.get_operations()[-1].name + ':0'

    return graph, input_name, output_name


def cal_jacobian(model, spectra, std, mean):
    """
    NAME: cal_jacobian
    PURPOSE: calculate jacobian
    INPUT:
    OUTPUT:
    HISTORY:
        2017-Nov-20 Henry Leung
    """

    tf_model, tf_input, tf_output = load_graph(model)

    x = tf_model.get_tensor_by_name(tf_input)

    # y = denormalize(tf_model.get_tensor_by_name(tf_output), std, mean)
    y = tf_model.get_tensor_by_name(tf_output)

    y_list = tf.unstack(y)
    num_outputs = y.shape.as_list()[0]

    print('It takes a long time for this operation to calculate jacobian')

    jacobian = np.empty((num_outputs, spectra.shape[0], spectra.shape[1]), dtype=np.float16)
    grads_wrt_input_tensor = [tf.gradients(y_element, x)[0] for y_element in y_list]
    with tf.Session(graph=tf_model) as sess:
        for i in range(spectra.shape[0]):
            jacobian[:, i:i + 1, :] = np.asarray(sess.run(grads_wrt_input_tensor,
                                                          feed_dict={x: spectra[i:i + 1]}))[:, :, :, 0]
            print(i)
    print('Completed operation of calculating jacobian')
    return jacobian


def prop_err(model, spectra, std, mean, err):
    """
    NAME: prop_err
    PURPOSE: calculate proporgation
    INPUT:
    OUTPUT: proporgated error
    HISTORY:
        2017-Nov-25 Henry Leung
    """

    spectra = spectra.reshape((spectra.shape[0], spectra.shape[1], 1))

    jac_matrix = cal_jacobian(model, spectra, std, mean)
    print('prop_eror')
    # for j in range(spectra.shape[0]):
    #     err = np.tile(err[j:j + 1], (jac_matrix[:, j].shape[1], 1))
    #     err[err > 3] = 0
    #     # covariance = np.einsum('ijk,kjl->jil', (jac_matrix[:,j] * (err[j:j+1] ** 2)), jac_matrix[:,j].T)
    #     temp = np.dot(err.T ** 2, (jac_matrix[:, j]).T)
    #     covariance = np.dot(jac_matrix[:, j], temp)
    err[err > 1] = 0
    covariance = np.einsum('ijk,kjl->jil', (jac_matrix * (err ** 2)), jac_matrix.T)
    print('\n')
    print('Finished')
    temp = np.diagonal(covariance, offset=0, axis1=1, axis2=2)
    temp = denormalize(temp, std, np.zeros(std.shape))
    print(temp)
    return temp


def jacobian(h5name=None, folder_name=None, number_spectra=100):
    """
    NAME: jacobian
    PURPOSE: calculate jacobian
    INPUT:
    OUTPUT:
    HISTORY:
        2017-Nov-20 Henry Leung
    """

    # prevent Tensorflow taking up all the GPU memory
    gpu_memory_manage()

    h5name_check(h5name)

    h5test = h5name + '_test.h5'
    traindata = h5name + '_train.h5'

    currentdir = os.getcwd()
    fullfolderpath = currentdir + '/' + folder_name
    mean_and_std = np.load(fullfolderpath + '/meanstd.npy')
    spec_meanstd = np.load(fullfolderpath + '/spectra_meanstd.npy')
    target = np.load(fullfolderpath + '/targetname.npy')
    model = keras_to_tf(folder_name=folder_name)

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
            # if tg == 'teff':
            #     print('yes')
            #     DR_fitlered_temp_upper = np.where(temp >= 5400)[0]
            #     index_not9999 = reduce(np.intersect1d, (index_not9999, DR_fitlered_temp_upper))
            else:
                index_not9999 = reduce(np.intersect1d, (index_not9999, temp_index))
        index_not9999 = index_not9999[0:number_spectra]

        test_spectra = np.array(F['spectra'])
        test_spectra_err = np.array(F['spectra_err'])
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

    spectra = test_spectra.reshape((number_spectra, test_spectra.shape[1], 1))
    jacobian = cal_jacobian(model, spectra, std_labels, mean_labels)

    jacobian = np.median(jacobian, axis=1)

    # Some plotting variables for asthetics
    plt.rcParams['axes.facecolor'] = 'white'
    sns.set_style("ticks")
    plt.rcParams['axes.grid'] = False
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.alpha'] = '0.4'
    path = os.path.join(fullfolderpath, 'jacobian')
    if not os.path.exists(path):
        os.makedirs(path)

    for j in range(num_labels):
        fullname = target_name_conversion(target[j])

        fig = plt.figure(figsize=(45, 30), dpi=150)
        dr = 14
        scale = np.max(np.abs((jacobian[j, :])))
        scale_2 = np.min((jacobian[j, :]))
        blue, green, red = chips_split(jacobian[j, :], dr=dr)
        lambda_blue, lambda_green, lambda_red = wavelegnth_solution(dr=dr)
        ax1 = fig.add_subplot(311)
        fig.suptitle('{}, Average of {} Stars'.format(fullname, number_spectra), fontsize=50)
        ax1.set_ylabel(r'$\partial$' + fullname, fontsize=40)
        ax1.set_ylim(scale_2, scale)
        ax1.plot(lambda_blue, blue, linewidth=0.9, label='astroNN')
        ax2 = fig.add_subplot(312)
        ax2.set_ylabel(r'$\partial$' + fullname, fontsize=40)
        ax2.set_ylim(scale_2, scale)
        ax2.plot(lambda_green, green, linewidth=0.9, label='astroNN')
        ax3 = fig.add_subplot(313)
        ax3.set_ylim(scale_2, scale)
        ax3.set_ylabel(r'$\partial$' + fullname, fontsize=40)
        ax3.plot(lambda_red, red, linewidth=0.9, label='astroNN')
        ax3.set_xlabel(r'Wavelength (Angstrom)', fontsize=40)

        ax1.axhline(0, ls='--', c='k', lw=2)
        ax2.axhline(0, ls='--', c='k', lw=2)
        ax3.axhline(0, ls='--', c='k', lw=2)

        try:
            if dr == 14:
                url = "https://svn.sdss.org/public/repo/apogee/idlwrap/trunk/lib/l31c/{}.mask".format(
                    aspcap_windows_url_correction(target[j]))
            else:
                raise ValueError('Only support DR14')
            df = np.array(pd.read_csv(urlopen(url), header=None, sep='\t'))
            print(url)
            aspcap_windows = df * scale
            aspcap_blue, aspcap_green, aspcap_red = chips_split(aspcap_windows, dr=dr)
            ax1.plot(lambda_blue, aspcap_blue, linewidth=0.9, label='ASPCAP windows')
            ax2.plot(lambda_green, aspcap_green, linewidth=0.9, label='ASPCAP windows')
            ax3.plot(lambda_red, aspcap_red, linewidth=0.9, label='ASPCAP windows')
        except:
            print('No ASPCAP windows data for {}'.format(aspcap_windows_url_correction(target[j])))
        tick_spacing = 50
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing / 1.5))
        ax3.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing / 1.7))
        ax1.minorticks_on()
        ax2.minorticks_on()
        ax3.minorticks_on()

        ax1.tick_params(labelsize=30, width=2, length=20, which='major')
        ax1.tick_params(width=2, length=10, which='minor')
        ax2.tick_params(labelsize=30, width=2, length=20, which='major')
        ax2.tick_params(width=2, length=10, which='minor')
        ax3.tick_params(labelsize=30, width=2, length=20, which='major')
        ax3.tick_params(width=2, length=10, which='minor')
        ax1.legend(loc='best', fontsize=40)
        plt.tight_layout()
        plt.subplots_adjust(left=0.05)
        plt.savefig(path + '/{}_jacobian.png'.format(target[j]))
        plt.close('all')
        plt.clf()
