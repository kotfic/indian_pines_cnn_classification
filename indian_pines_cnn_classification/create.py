import errno
import numpy as np
from sklearn.decomposition import PCA
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import random
from random import shuffle
from skimage.transform import rotate
import scipy.ndimage

from girder_worker.app import app
from girder_worker.utils import girder_job
from girder_worker_utils.decorators import task, task_input, task_output
from girder_worker_utils import types


def splitTrainTestSet(X, y, testRatio=0.10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=345,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def oversampleWeakClasses(X, y):
    uniqueLabels, labelCounts = np.unique(y, return_counts=True)
    maxCount = np.max(labelCounts)
    labelInverseRatios = maxCount / labelCounts
    # repeat for every label and concat
    newX = X[y == uniqueLabels[0], :, :, :].repeat(round(labelInverseRatios[0]), axis=0)
    newY = y[y == uniqueLabels[0]].repeat(round(labelInverseRatios[0]), axis=0)
    for label, labelInverseRatio in zip(uniqueLabels[1:], labelInverseRatios[1:]):
        cX = X[y== label,:,:,:].repeat(round(labelInverseRatio), axis=0)
        cY = y[y == label].repeat(round(labelInverseRatio), axis=0)
        newX = np.concatenate((newX, cX))
        newY = np.concatenate((newY, cY))
    np.random.seed(seed=42)
    rand_perm = np.random.permutation(newY.shape[0])
    newX = newX[rand_perm, :, :, :]
    newY = newY[rand_perm]
    return newX, newY


def standartizeData(X):
    newX = np.reshape(X, (-1, X.shape[2]))
    scaler = preprocessing.StandardScaler().fit(newX)
    newX = scaler.transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1],X.shape[2]))
    return newX, scaler

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createPatches(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def AugmentData(X_train):
    for i in range(int(X_train.shape[0]/2)):
        patch = X_train[i,:,:,:]
        num = random.randint(0,2)
        if (num == 0):

            flipped_patch = np.flipud(patch)
        if (num == 1):

            flipped_patch = np.fliplr(patch)
        if (num == 2):

            no = random.randrange(-180,180,30)
            flipped_patch = scipy.ndimage.interpolation.rotate(
                patch, no,axes=(1, 0), reshape=False, output=None,
                order=3, mode='constant', cval=0.0, prefilter=False)


    patch2 = flipped_patch
    X_train[i,:,:,:] = patch2

    return X_train



def loadIndianPinesData(data_path=None):
    if data_path is None:
        data_path = os.path.join(os.getcwd(),'data')

    data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
    labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']

    return data, labels



def savePreprocessedData(X_trainPatches, X_testPatches, y_trainPatches, y_testPatches, windowSize, data_path=None, wasPCAapplied = False, numPCAComponents = 0, testRatio = 0.25):

    if wasPCAapplied:
        if data_path is None:
            data_path = 'GITHUB'
        try:
            os.mkdir(data_path)
        except Exception as exc:
            if exc.errno != errno.EEXIST:
                raise

        with open(os.path.join(data_path, "XtrainWindowSize" + str(windowSize) + "PCA" + str(numPCAComponents) + "testRatio" + str(testRatio) + ".npy"), 'wb') as outfile:
            np.save(outfile, X_trainPatches)

        with open(os.path.join( data_path, "XtestWindowSize" + str(windowSize) + "PCA" + str(numPCAComponents) + "testRatio" + str(testRatio) + ".npy"), 'wb') as outfile:
            np.save(outfile, X_testPatches)
        with open(os.path.join( data_path, "ytrainWindowSize" + str(windowSize) + "PCA" + str(numPCAComponents) + "testRatio" + str(testRatio) + ".npy"), 'wb') as outfile:
            np.save(outfile, y_trainPatches)
        with open(os.path.join(data_path, "ytestWindowSize" + str(windowSize) + "PCA" + str(numPCAComponents) + "testRatio" + str(testRatio) + ".npy"), 'wb') as outfile:
            np.save(outfile, y_testPatches)
    else:
        if data_path is None:
            data_path = 'preprocessData'

        try:
            os.mkdir(data_path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

        with open(os.path.join(data_path, "XtrainWindowSize" + str(windowSize) + ".npy"), 'wb') as outfile:
            np.save(outfile, X_trainPatches)
        with open(os.path.join(data_path, "XtestWindowSize" + str(windowSize) + ".npy"), 'wb') as outfile:
            np.save(outfile, X_testPatches)
        with open(os.path.join(data_path, "ytrainWindowSize" + str(windowSize) + ".npy"), 'wb') as outfile:
            np.save(outfile, y_trainPatches)
        with open(os.path.join(data_path, "ytestWindowSize" + str(windowSize) + ".npy"), 'wb') as outfile:
            np.save(outfile, y_testPatches)

    return data_path



def preprocessData(data, labels, numComponents=30, windowSize=5, testRatio=0.25):
    X, y = data, labels
    X, pca = applyPCA(X,numComponents=numComponents)
    XPatches, yPatches = createPatches(X, y, windowSize=windowSize)
    X_train, X_test, y_train, y_test = splitTrainTestSet(XPatches, yPatches, testRatio)
    X_train, y_train = oversampleWeakClasses(X_train, y_train)
    X_train = AugmentData(X_train)

    return (X_train, X_test, y_train, y_test,
            {"windowSize": windowSize,
             "wasPCAapplied": True,
             "numPCAComponents": numComponents,
             "testRatio": testRatio})



# TODO: Fill in the function with the correct argument signature
# and code that performs the task.
@task()
@task_input('--data-path', type=types.Folder(), required=True, help='Path to input data')
@task_input('--output-path', type=types.Folder(), help='Path to save output data')
@task_input('--num-components', type=types.Integer(min=1), default=30, help='The number of components')
@task_input('--window-size', type=types.Integer(min=1), default=5, help='The window size')
@task_input('--test-ratio', type=types.Float(), default=0.25, help='The test ratio')
@task_output('path', type=types.Folder(), help='The path where output data is created')
@girder_job(title='Preprocess Data')
@app.task(bind=True)
def preprocess_data(self, data_path=None, output_path=None, num_components=30, window_size=5, test_ratio=0.25):
    """Preprocess data..."""
    data, labels = loadIndianPinesData(data_path=data_path)

    X_train, X_test, y_train, y_test, opts = preprocessData(
        data, labels,
        numComponents=num_components,
        windowSize=window_size,
        testRatio=test_ratio
    )

    return savePreprocessedData(X_train, X_test, y_train, y_test, data_path=output_path, **opts)


if __name__ == '__main__':
    preprocess_data.main()
