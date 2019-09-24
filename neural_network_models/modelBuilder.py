# -*- coding: utf-8 -*-
"""
    problem : "no name python in the module tensorflow"
    solution :
    -- in manage/settings
    -- in the search bar I pasted "python.linting.pylintEnable"
    -- and I unchecked the box "Wether to lint Python files using pylint"
"""
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.losses import mean_squared_error, categorical_crossentropy
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Add, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dropout, Dense

config = { "filters":128, "kernel_size":2, "strides":(1,1), "padding":"same", "kernel_regularizer":l2(1e-4)}

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

def model(name, nb_resnet=8):
    
    #common core
    in_x = x = Input((6,7,4))
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
