import argparse

from src.data.data_preprocessor import *
from src.inputs.data_preprocessor_input import *


def build_preprocessor_params(params):
    input = DataPreprocessorInput({
        'dataset_name': params.dataset_name,
        'split_path': params.split_path,
        'images_path': params.images_path,
        'final_size': params.final_size
    }).validate()

    dataset_name = input['dataset_name']
    split_path = input['split_path']
    images_path = input['images_path']
    final_size = tuple(input['final_size'])

    return dataset_name, split_path, images_path, final_size


def args_parser():
    parser = argparse.ArgumentParser(description='Preprocess the selected dataset')
    parser.add_argument('--dataset_name', type=str,
                        help='Dataset name that want to be preprocessed')
    parser.add_argument('--split_path', type=str,
                        help='Dataset split info path')
    parser.add_argument('--images_path', type=str,
                        help='Dataset images path')
    parser.add_argument('--final_size', nargs='+', type=int,
                        help='Final size of image (height, width)')
    return parser


if __name__ == '__main__':
    args = args_parser().parse_args()
    args = build_preprocessor_params(args)
    config = DataPreprocessorConfig(*args)
    preprocessor = DataPreprocessor(config).preprocess()
