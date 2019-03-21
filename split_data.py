import numpy as np
import pickle

from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit


def save_models_into_file(file_name, models):
    with open(file_name, 'wb') as file:
        pickle.dump(models, file)


def read_models_from_file(filename):
    file = open(filename, "rb")
    models = pickle.load(file)
    file.close()
    return models


if __name__ == '__main__':

    le = LabelEncoder()

    # features = read_models_from_file('Models/features.pckl')
    labels = read_models_from_file('Models/labels.pckl')

    encoded_labels = le.fit_transform(labels)
    one_hot_encoded_labels = to_categorical(encoded_labels, len(set(encoded_labels)))
    print(one_hot_encoded_labels)

    # sss_test = StratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=0)
    # for train_index, test_index in sss_test.split(features, encoded_labels):
    #     X_train, X_test = features[train_index], features[test_index]
    #     y_train, y_test = one_hot_encoded_labels[train_index], one_hot_encoded_labels[test_index]
    #     break
    #
    # save_models_into_file('Models/X_train.pckl', X_train)
    # save_models_into_file('Models/X_test.pckl', X_test)
    # save_models_into_file('Models/y_train.pckl', y_train)
    # save_models_into_file('Models/y_test.pckl', y_test)

