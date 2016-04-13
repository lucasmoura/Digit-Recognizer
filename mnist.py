import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf


DATA_FOLDER = 'data/'
TRAINING_DATA_PATH = DATA_FOLDER + 'train.csv'

TRAINING_DATA_PICKLE = DATA_FOLDER + 'data.pickle'


def apply_regularization(data, pixel_depth):
    mean = pixel_depth / 2
    for column in data:
        # TODO: Remove this check
        if column == 'label':
            continue

        data[column] = data[column].map(
            lambda x: (x - mean) / pixel_depth)

    return data


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(inputs, weights):
    return tf.nn.conv2d(
        inputs, weights, strides=[1, 1, 1, 1], padding='SAME')


def load_data():
    data = None

    with open(TRAINING_DATA_PICKLE, 'ra') as pkl_file:
        data = pickle.load(pkl_file)

    return data


def max_pool(inputs, window_size, stride):
    return tf.nn.max_pool(
        inputs, ksize=[1, window_size, window_size, 1],
        stride=[1, stride, stride, 1], padding='SAME')


def one_hot_encoding(label_array, num_labels):
    return (np.arange(num_labels) == label_array[:, None]).astype(np.float32)


def reformat_array(data_array, image_size, num_channels):
    data_array = data_array.reshape(
        -1, image_size, image_size, num_channels).astype(np.float32)
    return data_array


def save_data(data):
    with open(TRAINING_DATA_PICKLE, 'wa') as pkl_file:
        pickle.dump(data, pkl_file)


def shuffle_data(data):
    return data.reindex(np.random.permutation(data.index))


def split_data(data, split_ratio):
    data = shuffle_data(data)

    train_index = np.random.rand(len(data)) < split_ratio

    return (data[train_index], data[~train_index])


def split_labels_from_data(data, label):
    labels = data[label].to_frame(name=label)
    data = data.drop(label, 1)

    return (labels, data)


def weight_variable(shape, stdev=0.1):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def main():

    batch_size = 15
    conv1_kernels = 32
    depth = 16
    image_size = 28
    num_channels = 1
    num_labels = 10
    patch_size = 5
    pixel_depth = 255.0
    split_ratio = 0.9

    if os.path.isfile(TRAINING_DATA_PICKLE):
        print 'Loading data from pickle file...'
        mnist_data = load_data()
    else:
        print 'Reading csv...'
        mnist_data = pd.read_csv(TRAINING_DATA_PATH, header=0)
        print 'Applying feature regularization...'
        mnist_data = apply_regularization(mnist_data, pixel_depth)
        save_data(mnist_data)

    print 'Total data used: {}'.format(len(mnist_data))

    training_data, test_data = split_data(mnist_data, split_ratio)
    train_labels, training_data = split_labels_from_data(
        training_data, 'label')
    test_labels, test_data = split_labels_from_data(test_data, 'label')

    print 'Training data used: {}'.format(len(training_data))
    print 'Test data used: {}'.format(len(test_data))

    '''
    In order to use convnet in tensorflow, the input must be a
    4D array, with the following format: batch_size, image width,
    image height, number of channels. Therefore, the original array
    must be reshaped to fix that.
    '''
    training_array = reformat_array(
        training_data.as_matrix(), image_size, num_channels)
    test_array = reformat_array(
        test_data.as_matrix(), image_size, num_channels)

    '''
    Since this is a classification task and the softmax function will be
    used for the output neuron, the labels must be in a one hot enconding
    format
    '''
    train_labels = one_hot_encoding(train_labels.as_matrix(), num_labels)
    test_labels = one_hot_encoding(test_labels.as_matrix(), num_labels)

    graph = tf.Graph()

    with graph.as_default():

        '''
        For a weight for convnet must be a 4D array, that must contain:
        Both patch width and height, number of channels of the image and
        the number of kernels that will be used to capture features from
        the image.
        '''
        conv1_weight = weight_variable(
            [patch_size, patch_size, num_channels, conv1_kernels])
        conv1_bias = bias_variable([conv1_kernels])


if __name__ == '__main__':
    main()
