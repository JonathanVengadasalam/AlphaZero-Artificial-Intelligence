# -*- coding: utf-8 -*-
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.losses import mean_squared_error, categorical_crossentropy
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Add, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dropout, Dense

default_config = { "filters":128, "kernel_size":2, "strides":(1,1), "padding":"same", "kernel_regularizer":l2(1e-4)}

def _build_residual_block(x, config):
    in_x = x
    x = Conv2D(**config)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(**config)(x)
    x = BatchNormalization()(x)
    x = Add()([in_x, x])
    x = Activation("relu")(x)
    return x

def _build_normal_block(x, config):
    x = Conv2D(**config)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def build_model(name, height, width, depth, nb_resnet=8, **config):
    
    if config == {}: config = default_config
    
    #common core
    in_x = x = Input((height,width,depth))
    x = _build_normal_block(x, config)
    for _ in range(nb_resnet): x = _build_residual_block(x, config)
    res_out = x

    #policy branch
    x = Conv2D(filters=2, kernel_size=1, strides=(1,1), padding="same", kernel_regularizer=l2(1e-4))(res_out)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Flatten()(x)
    policy_out = Dense(7, kernel_regularizer=l2(1e-4), activation="softmax", name="policy_out")(x)

    #value branch
    x = Conv2D(filters=1, kernel_size=1, strides=(1,1), padding="same", kernel_regularizer=l2(1e-4))(res_out)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Flatten()(x)
    x = Dense(256, kernel_regularizer=l2(1e-4), activation="relu")(x)
    value_out = Dense(1, kernel_regularizer=l2(1e-4), activation="sigmoid", name="value_out")(x)

    #compile and save
    mod = Model(in_x, [policy_out, value_out], name=name)
    mod.compile(optimizer="sgd", loss=[categorical_crossentropy, mean_squared_error])
    mod.save(name)
    return mod
