import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from urllib.request import urlopen,urlretrieve
from PIL import Image
from tqdm import tqdm_notebook
from sklearn.utils import shuffle
import cv2

from keras.models import load_model, Sequential
from sklearn.datasets import load_files
from keras.utils import np_utils
from glob import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras.applications import resnet50
from keras.applications import VGG16

from VGG16.preprocessing import getdata
vgg_model = VGG16(include_top = False, weights="imagenet", input_shape = (256,256,3))

train_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

train_generator = train_datagen.flow_from_directory("../data/train",target_size=(256, 256),
        batch_size=32,
        class_mode=None)

val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory("../data/valid", target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

x, y = getdata("../data/train")
val_x, val_y = getdata("../data/valid")
vgg_features = vgg_model.predict(x)
vgg_val_features = vgg_model.predict(val_x)

print("************\n VGG features get. \n *************\n")

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=7 * 7 * 512))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=optimizers.adam(lr=2e-4),
              loss='binary_crossentropy',
              metrics=['acc'])
model.fit(vgg_features,
                    y,
                    epochs=20,
                    batch_size=8,
                    validation_data=(vgg_val_features,val_y))
model.save("model", include_optimizer=True)


