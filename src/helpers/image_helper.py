from dataclasses import astuple, dataclass
import cv2


@dataclass
class CropImageConfig:
    point_x: int
    point_y: int
    width: int
    height: int


def read(path):
    return cv2.imread(path)


def crop(image, config: CropImageConfig):
    point_x, point_y, width, height = astuple(config)
    new_height = point_y + height
    new_width = point_x + width
    cropped_image = image[point_y:new_height, point_x:new_width]
    return cropped_image


def show(image) -> None:
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(image, sizes):
    return cv2.resize(image, sizes)


def save(image, path) -> None:
    cv2.imwrite(path, image)
