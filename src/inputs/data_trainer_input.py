from src.inputs.base_input import BaseInput


class DataTrainerInput(BaseInput):
    def rules(self):
        rules = {
            'data_train_name': {
                'required': True,
                'type': 'string'
            },
            'data_test_name': {
                'required': True,
                'type': 'string'
            }
        }
        return rules
