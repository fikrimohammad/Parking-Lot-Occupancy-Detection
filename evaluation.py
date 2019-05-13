import numpy as np
import pickle

from keras.models import load_model

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def save_models_into_file(file_name, models):
    with open(file_name, 'wb') as file:
        pickle.dump(models, file)


def read_models_from_file(filename):
    file = open(filename, "rb")
    models = pickle.load(file)
    file.close()
    return models


if __name__ == '__main__':

    X_test = np.array(read_models_from_file('Models/CNR-EXT_x_camera7.pckl'))
    y_test = read_models_from_file('Models/CNR-EXT_y_camera7.pckl')

    checkpoint_path = 'Models\mAlexnet_trained_using_CNR-EXT_camera8.h5'

    model = load_model(checkpoint_path)

#     print(model.evaluate(X_test, y_test))

    prediction = model.predict(X_test)
    prediction = [np.argmax(x) for x in prediction]
    actual = [np.argmax(x) for x in y_test]

    print('\nPrecision :', round(precision_score(actual, prediction, average='weighted') * 100, 2))
    print('Recall :', round(recall_score(actual, prediction, average='weighted') * 100, 2))
    print('F1-Score :', round(f1_score(actual, prediction, average='weighted') * 100, 2))
    print('Accuracy :', round(accuracy_score(actual, prediction) * 100, 2))