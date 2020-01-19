import argparse

from src.data.data_preprocessor import DataPreprocessorConfig, DataPreprocessor
from src.inputs.data_preprocessor_input import DataPreprocessorInput


def build_preprocessor_params(params):
    validated_params = DataPreprocessorInput({
        'dataset_name': params.dataset_name,
        'split_path': params.split_path,
        'images_path': params.images_path,
        'final_size': params.final_size
    }).validate()

    dataset_name = validated_params['dataset_name']
    split_path = validated_params['split_path']
    images_path = validated_params['images_path']
    final_size = tuple(validated_params['final_size'])

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


def main():
    input_params = args_parser().parse_args()
    input_params = build_preprocessor_params(input_params)
    input_params = DataPreprocessorConfig(*input_params)
    DataPreprocessor(input_params).preprocess()


if __name__ == '__main__':
    main()
