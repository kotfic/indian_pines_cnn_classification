# Import the necessary libraries
from sklearn.decomposition import PCA
import os
import scipy.io as sio
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import spectral

from girder_worker.app import app
from girder_worker.utils import girder_job


def reports(model, X_test, y_test):
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
               ,'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
               'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
               'Stone-Steel-Towers']


    classification = classification_report(
        np.argmax(y_test, axis=1), y_pred, target_names=target_names)

    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    score = model.evaluate(X_test, y_test, batch_size=32)
    Test_Loss =  score[0]*100
    Test_accuracy = score[1]*100

    return classification, confusion, Test_Loss, Test_accuracy


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

def Patch(data,height_index,width_index, PATCH_SIZE=5):
    #transpose_array = data.transpose((2,0,1))
    #print transpose_array.shape
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]

    return patch


def loadIndianPinesData():
    data_path = os.path.join(os.getcwd(),'data')
    data = sio.loadmat(
        os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']

    labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']

    return data, labels

def loadTestData(windowSize=5, numPCAcomponents=30, testRatio=0.25):
    X_test = np.load(
        "GITHUB/XtestWindowSize"
        + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")
    y_test = np.load(
        "GITHUB/ytestWindowSize"
        + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")

    return X_test, y_test

def loadModel(path='my_model.h5'):
    return load_model(path)


def writeReport(Test_loss, Test_accuracy, classifciation, confusion, windowSize=5, numPCAcomponetns=30, testRatio=0.25):
    file_name = 'report' + "WindowSize" + str(windowSize) + "PCA" + str(numComponents) + "testRatio" + str(testRatio) +".txt"

    with open(file_name, 'w') as x_file:
        x_file.write('{} Test loss (%)'.format(Test_loss))
        x_file.write('\n')
        x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))


@girder_job(title="Validate Model")
@app.task
def validate(model, X_test, y_test):
    X_test  = np.reshape(X_test, (X_test.shape[0], X_test.shape[3], X_test.shape[1], X_test.shape[2]))
    y_test = np_utils.to_categorical(y_test)

    classification, confusion, Test_loss, Test_accuracy = reports(model, X_test,y_test)
    classification = str(classification)
    confusion = str(confusion)

    return Test_loss, Test_accuracy, classification, confusion

@girder_job(title="Classify Data")
@app.task
def classify(model, X, y, PATCH_SIZE = 5, numComponents = 30):
    X, pca = applyPCA(X,numComponents=numComponents)

    height = y.shape[0]
    width = y.shape[1]

    outputs = np.zeros((height,width))
    for i in range(height-PATCH_SIZE+1):
        for j in range(width-PATCH_SIZE+1):
            target = int(y[int(i+PATCH_SIZE/2), int(j+PATCH_SIZE/2)])
            if target == 0 :
                continue
            else :
                image_patch=Patch(X,i,j)
                #print (image_patch.shape)
                X_test_image = image_patch.reshape(
                    1, image_patch.shape[2],
                    image_patch.shape[0],
                    image_patch.shape[1]).astype('float32')

                prediction = (model.predict_classes(X_test_image))
                outputs[int(i+PATCH_SIZE/2)][int(j+PATCH_SIZE/2)] = prediction+1

    return outputs