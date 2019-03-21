import cv2
import numpy as np
import os
import pandas as pd

from tqdm import tqdm

def crop_img(img, x, y, w, h):
    x = int(round(x/2.592, 0))
    y = int(round(y/2.592, 0))
    w = int(round(w/2.592, 0))
    h = int(round(h/2.592, 0))
    cropped_img = img[y:y+h, x:x+w]
    return cropped_img


if __name__ == '__main__':

    full_image_path = "Dataset/FULL_IMAGE_1000x750/SUNNY/"

    for folder in tqdm(os.listdir(full_image_path)):
        for folder2 in os.listdir(full_image_path + folder):
            patch_info = pd.read_csv('Dataset/Patches_Information/' + folder2 + '.csv')
            for file in os.listdir(full_image_path + folder + '/' + folder2):
                img = cv2.imread(full_image_path + folder + '/' + folder2 + '/' + file)
                for i, row in patch_info.iterrows():
                    cropped_img = crop_img(img, row['X'], row['Y'], row['W'], row['H'])
                    cropped_img = cv2.resize(cropped_img, (150,150), interpolation = cv2.INTER_AREA)
                    output_path = 'Preprocessed_Dataset/' + folder2 + '/' + file[:-4] + '_' + str(row['SlotId']) + '_'  + '.jpg'
                    cv2.imwrite(output_path, cropped_img)