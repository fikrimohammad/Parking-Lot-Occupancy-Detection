from abc import abstractmethod
from datetime import datetime

from dataclasses import astuple

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

import src.configs.path as path

from src.helpers.file_helper import read_pckl


class BaseModelTrainer:
    def __init__(self):
        self.data_train_name = None
        self.model_name = type(self).__name__
        self.model = self.model_architecture()

    @abstractmethod
    def model_architecture(self):
        pass

    def train(self, params):
        x_train, x_test, y_train, y_test = self._load_data(params)

        checkpoint_path, logs_path, model_img_path = self._callbacks_path()
        early_stopper = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')
        checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir=logs_path)

        print(self.model.summary())
        plot_model(self.model, to_file=model_img_path, show_shapes=True, show_layer_names=True)

        self.model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(lr=0.001, decay=0.0005),
            metrics=['acc']
        )

        self.model.fit(
            x_train, y_train,
            batch_size=64,
            shuffle=False,
            epochs=18,
            validation_data=(x_test, y_test),
            callbacks=[checkpointer, early_stopper, tensorboard]
        )

    def _callbacks_path(self):
        today = datetime.now().strftime("%Y_%m_%d")
        base = '{}_{}_trained_using_{}'.format(
            today, self.model_name, self.data_train_name
        )
        checkpoint_path = '{}/{}.h5'.format(
            path.MODELS_PATH, base
        )
        logs_path = '{}/tensorboard_logs/{}'.format(
            path.REPORTS_PATH, base
        )
        model_img_path = '{}/figures/{}.png'.format(
            path.REPORTS_PATH, base
        )
        return checkpoint_path, logs_path, model_img_path

    def _load_data(self, params):
        data_train_name, data_test_name = astuple(params)

        self.data_train_name = data_train_name
        x_train_path = self._real_path('{}_x.pckl'.format(data_train_name))
        x_test_path = self._real_path('{}_x.pckl'.format(data_test_name))
        y_train_path = self._real_path('{}_y.pckl'.format(data_train_name))
        y_test_path = self._real_path('{}_y.pckl'.format(data_test_name))

        return read_pckl(x_train_path), read_pckl(x_test_path), \
               read_pckl(y_train_path), read_pckl(y_test_path)

    @staticmethod
    def _real_path(data_path):
        return '{}/{}'.format(path.PROCESSED_DATA_PATH, data_path)
