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

vgg_model = VGG16(include_top = False, weights="imagenet", input_shape = (256,256,3))

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)



generator = train_datagen.flow_from_directory(
    '../data/train',
    target_size=(256, 256),
    batch_size=32,
    class_mode=None,  # this means our generator will only yield batches of data, no labels
    shuffle=False)  # our data will be in order, so all first 1000 images will be cats, then 1000 dogs
# the predict_generator method returns the output of a model, given
# a generator that yields batches of numpy data
bottleneck_features_train = vgg_model.predict_generator(generator=generator, steps=5488 // 32)
# save the output as a Numpy array
np.save('bottleneck_features_train.npy', bottleneck_features_train)

test_datagen = ImageDataGenerator(rescale=1./255)
val_generator = test_datagen.flow_from_directory(
    '../data/valid',
    target_size=(256, 256),
    batch_size=32,
    class_mode=None,
    shuffle=False)
bottleneck_features_validation = vgg_model.predict_generator(val_generator, 670 // 32)
np.save('bottleneck_features_validation.npy', bottleneck_features_validation)
