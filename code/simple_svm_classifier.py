# -*- coding: utf-8 -*-
"""simple-svm-classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vBca3cOm2wTr-oUBKn-iK3BpQLinubx0
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

from __future__ import print_function
import numpy
import csv
import scipy.misc
import scipy
from scipy import ndimage
import pandas as pd
# import imutils
# import cv2
import numpy as np
from __future__ import print_function

# import dataprocessing

# from google.colab import drive
# drive.mount('/content/gdrive')

def flatten_matrix(matrix):
    vector = matrix.flatten(1)
    vector = vector.reshape(1, len(vector))
    return vector

def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
    U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
    epsilon = 0.1                #Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)                     #ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs)   #Data whitening

def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=True,
                              sqrt_bias=10, min_divisor=1e-8):
    assert X.ndim == 2, "X.ndim must be 2"
    scale = float(scale)
    assert scale >= min_divisor

    mean = X.mean(axis=1)
    if subtract_mean:
        X = X - mean[:, numpy.newaxis]
    else:
        X = X.copy()
    if use_std:
        ddof = 1
        if X.shape[1] == 1:
            ddof = 0
        normalizers = numpy.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
    else:
        normalizers = numpy.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale
    normalizers[normalizers < min_divisor] = 1.
    X /= normalizers[:, numpy.newaxis]  # Does not make a copy.
    return X
def ZeroCenter(data):
    data = data - numpy.mean(data,axis=0)
    return data

def normalize(arr):
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

def outputImage(pixels,number):
    data = pixels
    name = str(number)+"output.jpg"
    scipy.misc.imsave(name, data)

def Zerocenter_ZCA_whitening_Global_Contrast_Normalize(list):
    Intonumpyarray = numpy.asarray(list)
    data = Intonumpyarray.reshape(48,48)
    data2 = ZeroCenter(data)
    data3 = zca_whitening(flatten_matrix(data2)).reshape(48,48)
    data4 = global_contrast_normalize(data3)
    data5 = numpy.rot90(data4,3)
    return data5

def load_test_data():
    f = open('./fer2013.csv')
    csv_f = csv.reader(f)
    test_set_x =[]
    test_set_y =[]
    for row in csv_f:
        if str(row[2]) == "PrivateTest" :
            test_set_y.append(int(row[0]))
            temp_list = []
            for pixel in row[1].split( ):
                temp_list.append(int(pixel))
            data = Zerocenter_ZCA_whitening_Global_Contrast_Normalize(temp_list)
            test_set_x.append(data)
    return test_set_x, test_set_y

def load_data():

    train_x = []
    train_y = []
    val_x =[]
    val_y =[]
    test_x =[]
    test_y =[]

    with open("./badtrainingdata.txt", "r") as text:
        ToBeRemovedTrainingData = []
        for line in text:
            ToBeRemovedTrainingData.append(int(line))
    number = 0

    f = open('./fer2013.csv')
    csv_f = csv.reader(f)

    for row in csv_f:
        number+= 1
        if number not in ToBeRemovedTrainingData:

            if str(row[2]) == "Training" :
                temp_list = []

                for pixel in row[1].split( ):
                    temp_list.append(int(pixel))

                data = Zerocenter_ZCA_whitening_Global_Contrast_Normalize(temp_list)
                train_y.append(int(row[0]))
                train_x.append(data.reshape(2304).tolist())

            elif str(row[2]) == "PublicTest":
                temp_list = []

                for pixel in row[1].split( ):
                    temp_list.append(int(pixel))

                data = Zerocenter_ZCA_whitening_Global_Contrast_Normalize(temp_list)
                val_y.append(int(row[0]))
                val_x.append(data.reshape(2304).tolist())
            elif str(row[2]) == "PrivateTest":
                temp_list = []

                for pixel in row[1].split( ):
                    temp_list.append(int(pixel))

                data = Zerocenter_ZCA_whitening_Global_Contrast_Normalize(temp_list)
                test_y.append(int(row[0]))
                test_x.append(data.reshape(2304).tolist())

    return train_x, train_y, val_x, val_y, test_x, test_y

img_rows, img_cols = 48, 48
batch_size = 128
nb_classes = 7
nb_epoch = 250
img_channels = 1


Train_x, Train_y, Val_x, Val_y, Test_x, Test_y = load_data()

Train_x = numpy.asarray(Train_x)
# Train_x = Train_x.reshape(Train_x.shape[0],img_rows,img_cols)

Val_x = numpy.asarray(Val_x)
# Val_x = Val_x.reshape(Val_x.shape[0],img_rows,img_cols)

Test_x = numpy.asarray(Test_x)
# Test_x = Test_x.reshape(Test_x.shape[0],img_rows,img_cols)

# Train_x = Train_x.reshape(Train_x.shape[0], img_rows, img_cols,1)
# Val_x = Val_x.reshape(Val_x.shape[0], img_rows, img_cols,1)
# Test_x = Test_x.reshape(Test_x.shape[0], img_rows, img_cols,1)

Train_x = Train_x.astype('float32')
Val_x = Val_x.astype('float32')
Test_x = Test_x.astype('float32')


# Train_y = np_utils.to_categorical(Train_y, nb_classes)
# Val_y = np_utils.to_categorical(Val_y, nb_classes)
# Test_y = np_utils.to_categorical(Test_y, nb_classes)

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)
#fit to the trainin data
classifier.fit(Train_x,Train_y)

# now to Now predict the value of the digit on the test data
y_pred = classifier.predict(X_test)

print("Classification report for classifier %s:\n%s\n"
#       % (classifier, metrics.classification_report(y_test, y_pred)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_pred))
