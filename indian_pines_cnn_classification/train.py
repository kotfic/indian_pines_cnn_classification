from . import _quiet_import

import click
import numpy as np
import scipy
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.utils import np_utils

from girder_worker.app import app
from girder_worker.utils import girder_job
from girder_worker_utils.decorators import task, task_input, task_output
from girder_worker_utils import types


def loadTrainingDataset(data_path=None, windowSize = 5, numPCAcomponents = 30, testRatio = 0.25):
    if data_path is None:
        data_path = "GITHUB"

    X_train = np.load(os.path.join(
        data_path, "XtrainWindowSize"
        + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio)  + ".npy"))

    y_train = np.load(os.path.join(
        data_path,  "ytrainWindowSize"
        + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy"))

    return X_train, y_train



def saveModel(model, path = 'my_model.h5'):
    model.save(path)
    return path

def trainModel(X_train, y_train, windowSize=5, numPCAcomponents=30, testRatio=0.25):
    # Reshape into (numberofsumples, channels, height, width)
    X_train = np.reshape(X_train, (X_train.shape[0],
                                   X_train.shape[3],
                                   X_train.shape[1],
                                   X_train.shape[2]))

    # convert class labels to on-hot encoding
    y_train = np_utils.to_categorical(y_train)

    # Define the input shape
    input_shape= X_train[0].shape

    # number of filters
    C1 = 3*numPCAcomponents

    # Define the model
    model = Sequential()

    model.add(Conv2D(C1, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(3*C1, (3, 3), activation='relu'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(6*numPCAcomponents, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='softmax'))

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=32, epochs=15)

    return model


@task()
@task_input('--data-path', type=types.Folder(), default='GITHUB', help='Path to the input data')
@click.option('--model-path', type=click.STRING, default='my_model.h5', help='Where to save the model')
@task_input('--num-components', type=types.Integer(min=1), default=30, help='The number of components')
@task_input('--window-size', type=types.Integer(min=1), default=5, help='The window size')
@task_input('--test-ratio', type=types.Float(), default=0.25, help='The test ratio')
@task_output('path', type=types.File(), help='The path where the model is saved')
@girder_job(title="Train CNN")
@app.task
def train_model(data_path='GITHUB', model_path='my_model.h5', window_size=5, num_components=30, test_ratio=0.25):
    """Train the model"""
    X_train, y_train = loadTrainingDataset(data_path=data_path)

    model = trainModel(X_train, y_train,
                       windowSize=window_size,
                       numPCAcomponents=num_components,
                       testRatio=test_ratio)

    return saveModel(model, path=model_path)


if __name__ == '__main__':
    train_model.main()
