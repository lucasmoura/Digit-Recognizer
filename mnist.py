import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf


DATA_FOLDER = 'data/'
TRAINING_DATA_PATH = DATA_FOLDER + 'train.csv'
TEST_DATA_PATH = DATA_FOLDER + 'test.csv'

TRAINING_DATA_PICKLE = DATA_FOLDER + 'data.pickle'
TEST_DATA_PICKLE = DATA_FOLDER + 'test.pickle'

TRAINING_PARTITION_PICKLE = DATA_FOLDER + 'train_partition.pickle'
VALIDATION_PARTITION_PICKLE = DATA_FOLDER + 'validation_partition.pickle'
TEST_PARTITION_PICKLE = DATA_FOLDER + 'test_partition.pickle'

TRAINING_LABELS = DATA_FOLDER + 'train_labels.pickle'
VALIDATION_LABELS = DATA_FOLDER + 'validation_labels.pickle'
TEST_LABELS = DATA_FOLDER + 'test_labels.pickle'

MNIST_RESULT = DATA_FOLDER + 'result.csv'


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            predictions.shape[0])


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


def check_files():
    data_files = (os.path.isfile(TRAINING_PARTITION_PICKLE) and
                  os.path.isfile(VALIDATION_PARTITION_PICKLE) and
                  os.path.isfile(TEST_PARTITION_PICKLE))

    label_files = (os.path.isfile(TRAINING_LABELS) and
                   os.path.isfile(VALIDATION_LABELS) and
                   os.path.isfile(TEST_LABELS))

    return data_files and label_files


def conv2d(inputs, weights):
    return tf.nn.conv2d(
        inputs, weights, strides=[1, 1, 1, 1], padding='SAME')


def load_data(path):
    data = None

    with open(path, 'ra') as pkl_file:
        data = pickle.load(pkl_file)

    return data


def max_pool(inputs, window_size, stride):
    return tf.nn.max_pool(
        inputs, ksize=[1, window_size, window_size, 1],
        strides=[1, stride, stride, 1], padding='SAME')


def one_hot_encoding(label_array, num_labels):
    return (np.arange(num_labels) == label_array[:, None]).astype(np.float32)


def reformat_array(data_array, image_size, num_channels):
    data_array = data_array.reshape(
        -1, image_size, image_size, num_channels).astype(np.float32)
    return data_array


def save_data(data, path):
    with open(path, 'wa') as pkl_file:
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
    conv2_kernels = 64
    early_stopping = False
    image_size = 28
    num_channels = 1
    num_labels = 10
    num_neurons = 1024
    num_steps = 19000
    past_validation_error = 0
    patch_size = 5
    pixel_depth = 255.0
    pool_stride = 2
    pool_window = 2
    split_ratio = 0.8

    if os.path.isfile(TRAINING_DATA_PICKLE):
        print 'Loading data from pickle file...'
        mnist_data = load_data(TRAINING_DATA_PICKLE)
    else:
        print 'Reading data csv...'
        mnist_data = pd.read_csv(TRAINING_DATA_PATH, header=0)
        print 'Applying feature regularization...'
        mnist_data = apply_regularization(mnist_data, pixel_depth)
        save_data(mnist_data, TRAINING_DATA_PICKLE)

    if os.path.isfile(TEST_DATA_PICKLE):
        print 'Loading data from pickle file...'
        mnist_test_data = load_data(TEST_DATA_PICKLE)
    else:
        print 'Reading test csv...'
        mnist_test_data = pd.read_csv(TEST_DATA_PATH)
        print 'Applying feature regularization...'
        mnist_test_data = apply_regularization(mnist_test_data, pixel_depth)
        mnist_test_data = reformat_array(
            mnist_test_data.as_matrix(), image_size, num_channels)
        save_data(mnist_test_data, TEST_DATA_PICKLE)

    print 'Total data used: {}'.format(len(mnist_data))

    if check_files():
        print 'Loading training data...'
        training_data = load_data(TRAINING_PARTITION_PICKLE)
        print 'Loading validatio data...'
        validation_data = load_data(VALIDATION_PARTITION_PICKLE)
        print 'Loading test data...'
        test_data = load_data(TEST_PARTITION_PICKLE)

        print 'Loading train labels...'
        train_labels = load_data(TRAINING_LABELS)
        print 'Loading validation labels...'
        validation_labels = load_data(VALIDATION_LABELS)
        print 'Loading test labels...'
        test_labels = load_data(TEST_LABELS)
    else:
        training_data, test_data = split_data(mnist_data, split_ratio)
        training_data, validation_data = split_data(mnist_data, 0.8)
        train_labels, training_data = split_labels_from_data(
            training_data, 'label')
        test_labels, test_data = split_labels_from_data(test_data, 'label')
        validation_labels, validation_data = split_labels_from_data(
            validation_data, 'label')
        '''
        In order to use convnet in tensorflow, the input must be a
        4D array, with the following format: batch_size, image width,
        image height, number of channels. Therefore, the original array
        must be reshaped to fix that.
        '''
        training_data = reformat_array(
            training_data.as_matrix(), image_size, num_channels)
        validation_data = reformat_array(
            validation_data.as_matrix(), image_size, num_channels)
        test_data = reformat_array(
            test_data.as_matrix(), image_size, num_channels)

        print 'Saving training data...'
        save_data(training_data, TRAINING_PARTITION_PICKLE)
        print 'Saving validation data...'
        save_data(validation_data, VALIDATION_PARTITION_PICKLE)
        print 'Saving test data...'
        save_data(test_data, TEST_PARTITION_PICKLE)

        print 'Saving train labels...'
        save_data(train_labels, TRAINING_LABELS)
        print 'Saving validation labels...'
        save_data(validation_labels, VALIDATION_LABELS)
        print 'Saving test labels...'
        save_data(test_labels, TEST_LABELS)

    print 'Training data used: {}'.format(len(training_data))
    print 'Validation data used: {}'.format(len(validation_data))
    print 'Test data used: {}'.format(len(test_data))

    '''
    Since this is a classification task and the softmax function will be
    used for the output neuron, the labels must be in a one hot enconding
    format
    '''
    train_labels = one_hot_encoding(train_labels.as_matrix().T[0], num_labels)
    validation_labels = one_hot_encoding(validation_labels.as_matrix().T[0],
                                         num_labels)
    test_labels = one_hot_encoding(test_labels.as_matrix().T[0], num_labels)

    graph = tf.Graph()

    with graph.as_default():
        tf_train_data = tf.placeholder(
            tf.float32,
            shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(
            tf.float32, shape=(batch_size, num_labels))
        tf_mnist_test_data = tf.placeholder(
            tf.float32,
            shape=(280, image_size, image_size, num_channels))
        keep_prob = tf.placeholder(tf.float32)

        tf_valid_data = tf.constant(validation_data)
        tf_test_data = tf.constant(test_data)

        '''
        For a weight for convnet must be a 4D array, that must contain:
        Both patch width and height, number of channels of the image and
        the number of kernels that will be used to capture features from
        the image.
        '''
        conv1_weight = weight_variable(
            [patch_size, patch_size, num_channels, conv1_kernels])
        conv1_bias = bias_variable([conv1_kernels])

        conv2_weight = weight_variable(
            [patch_size, patch_size, conv1_kernels, conv2_kernels])
        conv2_bias = bias_variable([conv2_kernels])

        fc_weight1 = weight_variable([7 * 7 * 64, num_neurons])
        fc_bias1 = bias_variable([num_neurons])

        fc_weight2 = weight_variable([num_neurons, 10])
        fc_bias2 = bias_variable([10])

        def model(data):
            l_conv1 = tf.nn.relu(
                conv2d(data, conv1_weight) + conv1_bias)
            l_pool1 = max_pool(l_conv1, pool_window, pool_stride)

            l_conv2 = tf.nn.relu(conv2d(l_pool1, conv2_weight) + conv2_bias)
            l_pool2 = max_pool(l_conv2, pool_window, pool_stride)
            l_pool2_flat = tf.reshape(l_pool2, (-1, 7 * 7 * 64))

            l_fc1 = tf.nn.relu(tf.matmul(l_pool2_flat, fc_weight1) + fc_bias1)
            l_fc1_drop = tf.nn.dropout(l_fc1, keep_prob)

            return tf.matmul(l_fc1_drop, fc_weight2) + fc_bias2

        logits = model(tf_train_data)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_data))
        test_prediction = tf.nn.softmax(model(tf_test_data))
        mnist_test_prediction = tf.nn.softmax(model(tf_mnist_test_data))

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = training_data[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_data: batch_data,
                         tf_train_labels: batch_labels,
                         keep_prob: 0.5}

            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)

            if (step % 500 == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' %
                      accuracy(predictions, batch_labels))
                validation_accuracy = accuracy(
                    valid_prediction.eval(feed_dict={keep_prob: 1.0}),
                    validation_labels)
                print('Validation accuracy: %.1f%%' % validation_accuracy)

#                if not early_stopping:
#                    early_stopping = True
#                    past_validation_error = validation_accuracy
#                    continue
#
#                if past_validation_error > validation_accuracy:
#                    break
#                else:
#                    past_validation_error = validation_accuracy

        print('Test accuracy: %.1f%%' % accuracy(
            test_prediction.eval(feed_dict={keep_prob: 1.0}), test_labels))

        batch_size, count = 280, 0
        total_tests = mnist_test_data.shape[0]
        final_prediction = np.array([])

        while count != total_tests:
            batch_data = mnist_test_data[count: batch_size + count, :, :, :]
            result = mnist_test_prediction.eval(
                feed_dict={tf_mnist_test_data: batch_data, keep_prob: 1.0})
            final_prediction = np.concatenate(
                (final_prediction, np.argmax(result, 1)))
            count += batch_size

        print 'Saving mnist predictions...'
        count = 1
        with open(MNIST_RESULT, 'w') as result:
            result.write('"ImageId","Label"\n')
            for prediction in final_prediction:
                result.write('{},"{}"\n'.format(count, int(prediction)))
                count += 1


if __name__ == '__main__':
    main()
