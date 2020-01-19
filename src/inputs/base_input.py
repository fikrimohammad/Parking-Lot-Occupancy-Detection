import sys
from cerberus import Validator


class BaseInput:
    def __init__(self, params):
        self.validator = Validator(self.rules())
        self.params = params

    def validate(self):
        if self.validator.validate(self.params):
            return self.params
        self.__catch_errors(self.validator.errors)

    def rules(self):
        pass

    @staticmethod
    def __catch_errors(errors):
        print("Input parameter errors:")
        for key, value in errors.items():
            print("\t'{}': {}".format(key, value))
        sys.exit()
