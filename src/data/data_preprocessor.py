import numpy as np
import os
import src.config.path as path

from keras.utils import to_categorical
from src.helper.file_helper import *
from src.helper.image_helper import *
from tqdm import tqdm


@dataclass
class DataPreprocessorConfig:
    dataset_name: str
    split_folder: str
    img_folder: str
    final_img_size: tuple


class DataPreprocessor(object):

    def __init__(self, config: DataPreprocessorConfig):
        self.config = config
        self.features = []
        self.labels = []

    def preprocess(self):
        print("Preprocessing {} dataset".format(self.__dataset_name()))
        for filename in os.listdir(self.__split_folder()):
            self.__init_preprocess()
            infos = read_txt(self.__split_folder() + filename)
            print("Preprocessing {} split".format(filename))
            for info in tqdm(infos):
                info = info.split(" ")
                img_path = self.__img_folder() + info[0]
                img_label = info[1]

                img = self.__build_img(img_path)

                self.features.append(img)
                self.labels.append(img_label)

            features_path = '{}/{}_x_{}.pckl'.format(path.PROCESSED_DATA_PATH,
                                                     self.__dataset_name(),
                                                     filename[:-4])
            labels_path = '{}/{}_y_{}.pckl'.format(path.PROCESSED_DATA_PATH,
                                                   self.__dataset_name(),
                                                   filename[:-4])
            save_pckl(features_path, self.features)
            save_pckl(labels_path, to_categorical(self.labels))

    def __build_img(self, img_path):
        img = read(img_path)
        img = resize(img, self.__final_img_size())
        return img

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
