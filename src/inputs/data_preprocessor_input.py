from src.inputs.base_input import BaseInput


class DataPreprocessorInput(BaseInput):
    def rules(self):
        rules = {
            'dataset_name': {
                'required': True,
                'type': 'string'
            },
            'split_path': {
                'required': True,
                'type': 'string'
            },
            'images_path': {
                'required': True,
                'type': 'string'
            },
            'final_size': {
                'type': 'list',
                'schema': {
                    'type': 'integer',
                    'min': 1
                }
            }
        }
        return rules
