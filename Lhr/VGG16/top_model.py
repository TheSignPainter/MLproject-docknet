import numpy as np
import os
from PIL import Image
import cv2

from keras.models import load_model, Sequential
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.applications import VGG16
features = np.load("bottleneck_features_train.npy")
print(features.shape)

nb_train_docks = 1076
nb_train_nodocks = 4396
nb_validation_docks = 130
nb_validation_nodocks = 510
epochs = 50
batch_size = 16

top_model_weights_path = 'bottleneck_fc_model.h5'

def train_top_model():
    train_data = np.load('bottleneck_features_train.npy')
    train_labels = np.array(
        [0] * (nb_train_docks) + [1] * (nb_train_nodocks))

    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels = np.array(
        [0] * (nb_validation_docks) + [1] * (nb_validation_nodocks))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

train_top_model()
