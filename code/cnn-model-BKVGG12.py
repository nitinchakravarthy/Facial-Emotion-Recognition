from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.regularizers import l2 #, activity_l2
import numpy
import csv
import scipy.misc
import scipy
from scipy import ndimage
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import dataprocessing
def model_generate():
    	img_rows, img_cols = 48, 48
    	model = Sequential()
    	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 1), padding = 'same',bias_initializer='zeros', kernel_initializer='he_uniform'))
	model.add(Conv2D(32, (3, 3), activation='relu', padding = 'same',bias_initializer='zeros', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides =2))

	model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same',bias_initializer='zeros', kernel_initializer='he_uniform'))
	model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same',bias_initializer='zeros', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides =2))

	model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same',bias_initializer='zeros', kernel_initializer='he_uniform'))
	model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same',bias_initializer='zeros', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides =2))


	model.add(Conv2D(256, (3, 3), activation='relu', padding = "same",bias_initializer='zeros', kernel_initializer='he_uniform'))
	model.add(Conv2D(256, (3, 3), activation='relu', padding = "same",bias_initializer='zeros', kernel_initializer='he_uniform'))
	model.add(Conv2D(256, (3, 3), activation='relu', padding = "same",bias_initializer='zeros', kernel_initializer='he_uniform'))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(7))      
	model.add(Activation('softmax'))

    	ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    	model.compile(loss='categorical_crossentropy',
        	          optimizer=ada,
                	  metrics=['accuracy'])
    	model.summary()
    	return model
img_rows, img_cols = 48, 48
batch_size = 128
nb_classes = 7
nb_epoch = 250
img_channels = 1

Train_x, Train_y, Val_x, Val_y = dataprocessing.load_data()

Train_x = numpy.asarray(Train_x)
Train_x = Train_x.reshape(Train_x.shape[0],img_rows,img_cols)

Val_x = numpy.asarray(Val_x)
Val_x = Val_x.reshape(Val_x.shape[0],img_rows,img_cols)

Train_x = Train_x.reshape(Train_x.shape[0], img_rows, img_cols,1)
Val_x = Val_x.reshape(Val_x.shape[0], img_rows, img_cols,1)

Train_x = Train_x.astype('float32')
Val_x = Val_x.astype('float32')


Train_y = np_utils.to_categorical(Train_y, nb_classes)
Val_y = np_utils.to_categorical(Val_y, nb_classes)


model = model_generate()

filepath='model-vgg12-cnn.{epoch:02d}-{val_acc:.4f}.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(Train_x)

model.fit_generator(datagen.flow(Train_x, Train_y,
                    batch_size=batch_size),
                    samples_per_epoch=Train_x.shape[0],
                    nb_epoch=nb_epoch,
                    validation_data=(Val_x, Val_y),
                    callbacks=[checkpointer])
