import os
import pickle

import numpy as np
import pandas as pd


DATA_FOLDER = 'data/'
TRAINING_DATA_PATH = DATA_FOLDER + 'train.csv'

TRAINING_DATA_PICKLE = DATA_FOLDER + 'train.pickle'


def save_data(data):
    with open(TRAINING_DATA_PICKLE, 'wa') as pkl_file:
        pickle.dump(data, pkl_file)


def load_data():
    data = None

    with open(TRAINING_DATA_PICKLE, 'ra') as pkl_file:
        data = pickle.load(pkl_file)

    return data


def shuffle_data(data):
    return data.reindex(np.random.permutation(data.index))


def split_data(data, split_ratio):
    data = shuffle_data(data)

    train_index = np.random.rand(len(data)) < split_ratio

    return (data[train_index], data[~train_index])


def main():

    split_ratio = 0.9

    if os.path.isfile(TRAINING_DATA_PICKLE):
        print 'Loading data from pickle file...'
        mnist_data = load_data()
    else:
        print 'Reading csv...'
        mnist_data = pd.read_csv(TRAINING_DATA_PATH, header=0)
        save_data(mnist_data)

    print 'Total data used: {}'.format(len(mnist_data))

    training_data, test_data = split_data(mnist_data, split_ratio)

    print 'Training data used: {}'.format(len(training_data))
    print 'Test data used: {}'.format(len(test_data))


if __name__ == '__main__':
    main()
