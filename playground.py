import cv2
import numpy as np
import os
import pandas as pd

from keras.models import load_model

from tqdm import tqdm


def crop_img(img, x, y, w, h):
    x = int(round(x / 2.592, 0))
    y = int(round(y / 2.592, 0))
    w = int(round(w / 2.592, 0))
    h = int(round(h / 2.592, 0))
    cropped_img = img[y:y + h, x:x + w]
    return cropped_img


def draw_rectangle(image, x, y, w, h, label, slotId):
    x = int(round(x / 2.592, 0))
    y = int(round(y / 2.592, 0))
    w = int(round(w / 2.592, 0))
    h = int(round(h / 2.592, 0))
    if label == 1:
        cv2.rectangle(image, (x, y), (x + h, y + w), (0, 0, 255), 2)
    else:
        cv2.rectangle(image, (x, y), (x + h, y + w), (0, 255, 0), 2)

    cv2.putText(image, "ID:" + str(slotId), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    return


def predict(model, features):
    prediction = model.predict(np.expand_dims(features, axis=0))
    prediction = np.argmax(prediction)

    return prediction


if __name__ == '__main__':

    full_image_path = "D:\\Tugas Kuliah\\Deep Learning\\Parking Lot Occupancy Detection\\Dataset Mentah\\FULL_IMAGE_1000x750\\RAINY\\"

    model_path = "Models/mAlexnet_trained_using_CNR-EXT_camera8.h5"
    model = load_model(model_path)

    for folder in os.listdir(full_image_path):
        for folder2 in tqdm(os.listdir(full_image_path + folder)):
            if folder2.startswith('camera8'):
                patch_info = pd.read_csv(
                    'D:\\Tugas Kuliah\\Deep Learning\\Parking Lot Occupancy Detection\\Dataset Mentah\\Patches_Information\\' + folder2 + '.csv')
                for file in os.listdir(full_image_path + folder + '/' + folder2):
                    img = cv2.imread(full_image_path + folder + '\\' + folder2 + '\\' + file)
                    for i, row in patch_info.iterrows():
                        cropped_img = crop_img(img, row['X'], row['Y'], row['W'], row['H'])
                        cropped_img = cv2.resize(cropped_img, (224, 224), interpolation=cv2.INTER_AREA)
                        label = predict(model, cropped_img)
                        draw_rectangle(img, row['X'], row['Y'], row['W'], row['H'], label, row['SlotId'])
                        # cv2.rectangle(img, (row['X'], row['Y']), (row['X'] + row['H'], row['Y'] + row['W']), (255, 0, 0), 2)
                    cv2.imshow("Tempat Parkir", img)
                    k = cv2.waitKey(0)
                break
            else:
                continue
        break
