import sys
from cerberus import Validator


class BaseInput:
    def __init__(self, input):
        self.validator = Validator(self.rules())
        self.input = input

    def validate(self):
        if self.validator.validate(self.input):
            return self.input
        else:
            self.__catch_errors(self.validator.errors)

    def rules(self):
        pass

    @staticmethod
    def __catch_errors(errors):
        sys.exit(errors)
