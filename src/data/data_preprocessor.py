import os

from dataclasses import dataclass
import numpy as np

from keras.utils import to_categorical
from tqdm import tqdm

import src.configs.path as path
from src.helpers.file_helper import read_txt, save_pckl
from src.helpers.image_helper import read, resize


@dataclass
class DataPreprocessorConfig:
    dataset_name: str
    split_folder: str
    img_folder: str
    final_img_size: tuple


class DataPreprocessor:
    def __init__(self, config: DataPreprocessorConfig):
        self.config = config
        self.features = []
        self.labels = []

    def preprocess(self):
        print("Preprocessing {} dataset".format(self.__dataset_name()))
        for filename in os.listdir(self.__split_folder()):
            self.__init_preprocess()
            img_infos = read_txt(self.__split_folder() + filename)
            print("Preprocessing {} split".format(filename))
            for img_info in tqdm(img_infos):
                self.__build_img(img_info)
            self.__finalization(filename)

    def __build_img(self, info):
        info = info.split(" ")
        img_path = self.__img_folder() + info[0]
        img_label = info[1]

        img = read(img_path)
        img = resize(img, self.__final_img_size())

        self.features.append(img)
        self.labels.append(img_label)

    def __finalization(self, filename):
        features_path = '{}/{}_x_{}.pckl'.format(path.PROCESSED_DATA_PATH,
                                                 self.__dataset_name(),
                                                 filename[:-4])
        labels_path = '{}/{}_y_{}.pckl'.format(path.PROCESSED_DATA_PATH,
                                               self.__dataset_name(),
                                               filename[:-4])
        save_pckl(features_path, np.array(self.features))
        save_pckl(labels_path, to_categorical(self.labels))

    def __dataset_name(self) -> str:
        return self.config.dataset_name

    def __split_folder(self) -> str:
        return '{}/{}'.format(path.RAW_DATA_PATH,
                              self.config.split_folder)

    def __img_folder(self) -> str:
        return '{}/{}'.format(path.RAW_DATA_PATH,
                              self.config.img_folder)

    def __final_img_size(self) -> tuple:
        return self.config.final_img_size

    def __init_preprocess(self) -> None:
        self.features = []
        self.labels = []
