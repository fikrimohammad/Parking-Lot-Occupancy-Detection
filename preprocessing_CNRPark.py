import cv2
import numpy as np
import os
import pickle

from keras.utils import to_categorical

from tqdm import tqdm


def save_models_into_file(file_name, models):
    with open(file_name, 'wb') as file:
        pickle.dump(models, file)


if __name__ == '__main__':

    for filename in os.listdir('CNRPark/Splits'):
        file = open('CNRPark/Splits/' + filename, 'r', encoding='utf-8').readlines()
        features = []
        labels = []
        for x in tqdm(file):
            tmp = x.split(' ')
            image_label = tmp[1]
            image_path = "CNRPark/" + tmp[0]

            image = cv2.imread(image_path)
            image = cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)

            features.append(image)
            labels.append(image_label)

        features_path = 'Models/CNRPark_x_' + filename[:-4] + '.pckl'
        labels_path = 'Models/CNRPark_y_' + filename[:-4] + '.pckl'

        save_models_into_file(features_path, features)
        save_models_into_file(labels_path, to_categorical(labels))
