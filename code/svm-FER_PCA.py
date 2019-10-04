#!/usr/bin/env python
# coding: utf-8

# In[9]:



import numpy
import csv
import scipy.misc
import scipy
from scipy import ndimage
import pandas as pd

import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.externals import joblib

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
# print('staring svm training')
# # Create a classifier: a support vector classifier
# classifier = svm.SVC(gamma=0.001)
# #fit to the trainin data
# classifier.fit(Train_x,Train_y)
# print('finished svm training')


# In[4]:



# now to Now predict the value of the digit on the test data
# y_pred = classifier.predict(Test_x)

# print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(Test_y, y_pred)))

# print("Confusion matrix:\n%s" % metrics.confusion_matrix(Test_y, y_pred))


# In[5]:




# # Save to file in the current working directory
# pkl_filename = "simple_svm_model.pkl"  
# with open(pkl_filename, 'wb') as file:  
#     joblib.dump(classifier, file)

# # Load from file
# with open(pkl_filename, 'rb') as file:  
#     pickle_model = joblib.load(file)

# # Calculate the accuracy score and predict target values
# score = pickle_model.score(Test_x, Test_y)  
# print("pickle Test score: {0:.2f} %".format(100 * score))  
# # Ypredict = pickle_model.predict(Test_x) 


# In[16]:


n_components = 200
h = 48
w =48
from sklearn.decomposition import PCA

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, len(Train_x)))
# t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(Train_x)
# print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
# t0 = time()
X_train_pca = pca.transform(Train_x)
X_test_pca = pca.transform(Test_x)
# print("done in %0.3fs" % (time() - t0))


# In[ ]:


# print('staring svm training')
# # Create a classifier: a support vector classifier
# classifier = svm.SVC(gamma=0.001)
# #fit to the trainin data
# classifier.fit(X_train_pca,Train_y)
# print('finished svm training')

from sklearn.model_selection import GridSearchCV

print("Fitting the classifier to the actual training set")
# t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf_actual = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5)
clf_actual = clf_actual.fit(Train_x, Train_y)
# print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf_actual.best_estimator_)
print(clf)


# In[15]:


y_pred = classifier.predict(X_test_pca)

print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(Test_y, y_pred)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(Test_y, y_pred))


# In[ ]:







# Train a SVM classification model on PCA reshaped data

# print("Fitting the classifier to the PCA reshaped training set")
# t0 = time()
# param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
# clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
#                    param_grid, cv=5)
# clf = clf.fit(X_train_pca, y_train)
# print("done in %0.3fs" % (time() - t0))
# print("Best estimator found by grid search:")
# print(clf.best_estimator_)


# In[ ]:


# Train a SVM classification model on actual data

# print("Fitting the classifier to the actual training set")
# t0 = time()
# param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
# clf_actual = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
#                    param_grid, cv=5)
# clf_actual = clf_actual.fit(Train_x, Train_y)
# print("done in %0.3fs" % (time() - t0))
# print("Best estimator found by grid search:")
# print(clf.best_estimator_)




# now to Now predict the value of the digit on the test data
# y_pred = classifier.predict(Test_x)

# print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(Test_y, y_pred)))

# print("Confusion matrix:\n%s" % metrics.confusion_matrix(Test_y, y_pred))



# Calculate the accuracy score and predict target values
print('Before saving')
score = clf_actual.score(Test_x, Test_y)  
print("pickle Test score: {0:.2f} %".format(100 * score))  
# Ypredict = pickle_model.predict(Test_x) 

print('saving')

# Save to file in the current working directory
pkl_filename = "grid-search-pca-svm.pkl"  
with open(pkl_filename, 'wb') as file:  
    joblib.dump(clf_actual, file)
print('After saving')

# Load from file
with open(pkl_filename, 'rb') as file:  
    pickle_model = joblib.load(file)

# Calculate the accuracy score and predict target values
score = pickle_model.score(Test_x, Test_y)  
print("pickle Test score: {0:.2f} %".format(100 * score))  
# Ypredict = pickle_model.predict(Test_x) 

