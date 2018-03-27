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


def loadTrainingDataset(windowSize = 5, numPCAcomponents = 30, testRatio = 0.25):
    X_train = np.load(
        "GITHUB/XtrainWindowSize"
        + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio)  + ".npy")

    y_train = np.load(
        "GITHUB/ytrainWindowSize"
        + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")

    return X_train, y_train



def saveModel(model, path = 'my_model.h5'):
    model.save('my_model.h5')

@girder_job(title="Train CNN")
@app.task
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
    print(input_shape)

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
