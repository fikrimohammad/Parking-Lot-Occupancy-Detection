import cv2
import numpy as np
import os
import pickle

from tqdm import tqdm


def save_models_into_file(file_name, models):
    with open(file_name, 'wb') as file:
        pickle.dump(models, file)


if __name__ == '__main__':

    root_path = 'CNRPark/'

    features = []
    labels = []

    for folder in os.listdir(root_path):
        for folder2 in os.listdir(root_path + folder):
            for file in tqdm(os.listdir(root_path + folder + '/' + folder2)):
                img_path = root_path + folder + '/' + folder2 + '/' + file
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)

                features.append(img)
                labels.append(folder2)

    save_models_into_file('Models/features.pckl', np.array(features))
    save_models_into_file('Models/labels.pckl', np.array(labels))