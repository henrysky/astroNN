# ---------------------------------------------------------#
#   astroNN.NN.error_eval: evaluate error in NN model
# ---------------------------------------------------------#

import os
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
import numpy as np


def keras_to_tf(weight_file, input_fld='', output_fld=''):

    # weight_file is a .h5 keras model file
    output_node_names_of_input_network = ["pred0"]
    output_node_names_of_final_network = 'output_node'
    output_graph_name = weight_file[:-2] + 'pb'
    weight_file_path = os.path.join(input_fld, weight_file)

    net_model = load_model(weight_file_path)

    num_output = len(output_node_names_of_input_network)
    pred = [None] * num_output
    pred_node_names = [None] * num_output
    for i in range(num_output):
        pred_node_names[i] = output_node_names_of_final_network + str(i)
        pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])

    sess = K.get_session()

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, output_fld, output_graph_name, as_text=False)
    print('saved the constant graph (ready for inference) at: ', os.path.join(output_fld, output_graph_name))

    return output_fld + output_graph_name


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

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


def compute_jacobian_from_tf_model_path(tf_model_path, input_data, denormalize):
    tf_model, tf_input, tf_output = load_graph(tf_model_path)

    x = tf_model.get_tensor_by_name(tf_input)

    y = denormalize(tf_model.get_tensor_by_name(tf_output))

    y_list = tf.unstack(y)
    num_outputs = y.shape.as_list()[0]

    if input_data.shape[0] == 1:
        with tf.Session(graph=tf_model) as sess:
            y_out = sess.run([tf.gradients(y_, x)[0] for y_ in y_list], feed_dict={
                x: input_data
            })
            jacobian = np.asarray(y_out)
            jacobian = jacobian[:, :, :, 0]
    else:
        print('\nCreating jacobian matrices for ' + str(len(input_data)) + ' spectra...\n')
        print_count = int(len(input_data) / 10)
        if print_count == 0:
            print_count = 1

        jacobian = np.zeros((num_outputs, input_data.shape[0], input_data.shape[1]))
        for i in range(input_data.shape[0]):
            with tf.Session(graph=tf_model) as sess:
                y_out = sess.run([tf.gradients(y_, x)[0] for y_ in y_list], feed_dict={
                    x: input_data[i:i + 1]
                })
            jac_temp = np.asarray(y_out)
            jacobian[:, i:i + 1, :] = jac_temp[:, :, :, 0]
            if (i + 1) % print_count == 0:
                print(str(i + 1) + ' jacobians completed...\n')
        print('All ' + str(i + 1) + ' jacobians completed.\n')
    return jacobian


def compute_covariance_from_tf_model_path(tf_model_path,input_data,var,denormalize=None):
    jac_matrix = compute_jacobian_from_tf_model_path(tf_model_path,input_data,denormalize)
    var[var > 6] = 0
    jac_matrix = np.nan_to_num(jac_matrix)
    covariance = np.einsum('ijk,kjl->jil',(jac_matrix*(var**2)),jac_matrix.T)
    return covariance


def compute_variance_from_tf_model_path(tf_model_path,input_data,var,denormalize=None):
    covariance = compute_covariance_from_tf_model_path(tf_model_path,input_data,var,denormalize)
    return np.diagonal(covariance, offset=0, axis1=1, axis2=2)