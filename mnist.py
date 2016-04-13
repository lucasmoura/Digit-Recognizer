import os
import pandas as pd
import pickle


DATA_FOLDER = 'data/'
TRAINING_DATA_PATH = DATA_FOLDER + 'train.csv'

TRAINING_DATA_PICKLE = DATA_FOLDER + 'train.pickle'


def load_data():
    data = None

    with open(TRAINING_DATA_PICKLE, 'ra') as pkl_file:
        data = pickle.load(pkl_file)

    return data


def save_data(data):
    with open(TRAINING_DATA_PICKLE, 'wa') as pkl_file:
        pickle.dump(data, pkl_file)


def main():

    if os.path.isfile(TRAINING_DATA_PICKLE):
        training_data = load_data()
    else:
        training_data = pd.read_csv(TRAINING_DATA_PATH, header=0)
        save_data(training_data)


if __name__ == '__main__':
    main()
