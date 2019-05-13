import joblib
import numpy as np
import pickle

from time import time

from keras import Model
from keras.layers import Activation, BatchNormalization, Convolution2D, Dense, Flatten, Input, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils.vis_utils import plot_model


def save_models_into_file(file_name, models):
    with open(file_name, 'wb') as file:
        pickle.dump(models, file)


def read_models_from_file(filename):
    file = open(filename, "rb")
    models = pickle.load(file)
    file.close()
    return models


if __name__ == '__main__':

    X_train = np.array(read_models_from_file('Models/CNR-EXT_x_camera7.pckl'))
    X_test = np.array(read_models_from_file('Models/CNR-EXT_x_camera8.pckl'))
    y_train = read_models_from_file('Models/CNR-EXT_y_camera7.pckl')
    y_test = read_models_from_file('Models/CNR-EXT_y_camera8.pckl')

    checkpoint_path = 'Models/mAlexnet_trained_using_CNR-EXT_camera7.h5'
    early_stopper = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir="logs/mAlexnet_trained_using_CNR-EXT_camera7_{}".format(time()))

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

    model_output = Dense(len(y_train[0]), activation='softmax')(z)

    model = Model(model_input, model_output)

    print(model.summary())
    plot_model(model, to_file='mAlexnet_trained_using_CNR-EXT_camera7.png', show_shapes=True, show_layer_names=True)

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001, decay=0.0005),
                  metrics=['acc'])

    # Fit the model
    model.fit(X_train, y_train,
              batch_size=64,
              shuffle=False,
              epochs=18,
              validation_data=(X_test, y_test),
              callbacks=[checkpointer, early_stopper, tensorboard])