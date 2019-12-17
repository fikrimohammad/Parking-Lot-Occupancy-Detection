import argparse

from src.data.data_preprocessor import *


def args_parser():
    parser = argparse.ArgumentParser(description='Preprocess the selected dataset')
    parser.add_argument('--dataset_name', type=str,
                        help='Dataset name that want to be preprocessed')
    parser.add_argument('--split_path', type=str,
                        help='Dataset split info path')
    parser.add_argument('--images_path', type=str,
                        help='Dataset split images path')
    parser.add_argument('--final_size', nargs='+', type=int,
                        help='Final size of image (height, width)')
    return parser


if __name__ == '__main__':
    args = args_parser().parse_args()

    dataset_name = args.dataset_name
    split_path = args.split_path
    images_path = args.images_path
    final_size = tuple(args.final_size)

    config = DataPreprocessorConfig(dataset_name, split_path, images_path, final_size)
    preprocessor = DataPreprocessor(config).preprocess()
