from src.models.base_model_trainer import BaseModelTrainer

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Activation, BatchNormalization, Convolution2D, Dense, Flatten, MaxPooling2D


class mAlexNet(BaseModelTrainer):
    def model_architecture(self):
        model_input = Input(shape=(224, 224, 3))

        z = Convolution2D(filters=16, kernel_size=11, strides=(4, 4), padding='valid')(model_input)
        z = Activation('relu')(z)
        z = BatchNormalization()(z)
        z = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(z)

        z = Convolution2D(filters=20, kernel_size=5, strides=(1, 1), padding='valid')(z)
        z = Activation('relu')(z)
        z = BatchNormalization()(z)
        z = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(z)

        z = Convolution2D(filters=30, kernel_size=3, strides=(1, 1), padding='valid')(z)
        z = Activation('relu')(z)
        z = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(z)

        z = Flatten()(z)
        z = Dense(48, activation='relu')(z)

        model_output = Dense(2, activation='softmax')(z)

        return Model(model_input, model_output)
