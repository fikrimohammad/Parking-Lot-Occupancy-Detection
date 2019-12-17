import cv2
import numpy as np
from dataclasses import astuple, dataclass


@dataclass
class CropImageConfig:
    point_x: int
    point_y: int
    width: int
    height: int


def read(path):
    return cv2.imread(path)


def crop(image, config: CropImageConfig):
    x, y, w, h = astuple(config)
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image


def show(image) -> None:
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(image, sizes):
    return cv2.resize(image, sizes)


def save(image, path) -> None:
    cv2.imwrite(path, image)
