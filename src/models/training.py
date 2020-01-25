import argparse

from src.inputs.data_trainer_input import DataTrainerInput

from src.models.mAlexNet import mAlexNet

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    data_train_feature_name: str
    data_test_feature_name: str


def build_model_trainer_params(params):
    validated_params = DataTrainerInput({
        'data_train_name': params.data_train_name,
        'data_test_name': params.data_test_name,
    }).validate()

    data_train_name = validated_params['data_train_name']
    data_test_name = validated_params['data_test_name']

    return data_train_name, data_test_name


def args_parser():
    parser = argparse.ArgumentParser(description='Train preprocessed data to become machine learning model')
    parser.add_argument('--data_train_name', type=str,
                        help='Data train name')
    parser.add_argument('--data_test_name', type=str,
                        help='Data test name')
    return parser


if __name__ == '__main__':
    input_params = args_parser().parse_args()
    input_params = build_model_trainer_params(input_params)
    input_params = ModelTrainerConfig(*input_params)
    mAlexNet().train(input_params)
