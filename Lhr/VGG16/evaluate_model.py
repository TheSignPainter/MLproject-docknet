import keras.models
from keras.preprocessing.image import ImageDataGenerator
import os

model = keras.models.load_model("model_joint.h5")

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_data_dir = '../../data/valid'
img_width, img_height = 256, 256
batch_size = 16

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

model.evalaute_generator(test_generator, verbose = 0)