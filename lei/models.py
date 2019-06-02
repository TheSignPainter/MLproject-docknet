import logging

import keras.backend as K
import math
from keras import layers
from keras import regularizers
from keras.layers import Input, GRU, AveragePooling2D, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.core import Lambda, Dense, RepeatVector
from keras.layers.core import Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import tensorflow as tf

#from constants import *
#from attack_utils import add_noise, fbank_layer
def clipped_relu(inputs):
    return Lambda(lambda y: K.minimum(K.maximum(y, 0), 20))(inputs)


def identity_block(input_tensor, kernel_size, filters, stage, block):
    conv_name_base = 'res{}_{}_branch'.format(stage, block)

    x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.00001),
                   name=conv_name_base + '_2a')(input_tensor)
    x = BatchNormalization(name=conv_name_base + '_2a_bn')(x)
    x = clipped_relu(x)

    x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.00001),
                   name=conv_name_base + '_2b')(x)
    x = BatchNormalization(name=conv_name_base + '_2b_bn')(x)


    x = layers.add([x, input_tensor])
    x = clipped_relu(x)
    return x




def convolutional_model(input_shape,    #input_shape(256,256,3)
                        batch_size):
    # http://cs231n.github.io/convolutional-networks/
    # conv weights
    # #params = ks * ks * nb_filters * num_channels_input


    def conv_and_res_block(inp, filters, stage):
        conv_name = 'conv{}-s'.format(filters)
        o = Conv2D(filters,
                       kernel_size=5,
                       strides=2,
                       padding='same',
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=regularizers.l2(l=0.00001), name=conv_name)(inp)
        o = BatchNormalization(name=conv_name + '_bn')(o)
        o = clipped_relu(o)
        for i in range(3):
            o = identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i)
        return o

    def cnn_component(inp):
        x_ = conv_and_res_block(inp, 64, stage=1)
        x_ = conv_and_res_block(x_, 128, stage=2)
        x_ = conv_and_res_block(x_, 256, stage=3)
        x_ = conv_and_res_block(x_, 512, stage=4)
        return x_

    inputs = Input(shape=input_shape)  # TODO the network should be definable without explicit batch shape
    #x = Lambda(lambda y: K.reshape(y, (batch_size*num_frames,input_shape[1], input_shape[2], input_shape[3])), name='pre_reshape')(inputs)
    x = cnn_component(inputs)  # .shape = (BATCH_SIZE , num_frames/16, 64/16, 512)
    
    x = AveragePooling2D(pool_size = (2, 2,), padding = 'valid')(x) 

    x = Flatten()(x)
    x = Dense(1, name = 'affine2', activation = 'sigmoid')(x)
    #x = Reshape((-1,2048))(x)
    #x = Lambda(lambda y: K.reshape(y, (-1, math.ceil(num_frames/16), 2048)), name='reshape')(x)
    #x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x)  #shape = (BATCH_SIZE, 512)
    #x = Dense(512, name='affine')(x)  # .shape = (BATCH_SIZE , 512)
    #x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)
    
    model = Model(inputs, x, name='convolutional')

    #print(model.summary())
    return model

input_shape = (256,256, 3)
batch_size = 1
model = convolutional_model(input_shape, batch_size)
print(model.output.shape)
