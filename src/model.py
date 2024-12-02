from keras.layers import (
    Input,
    Dense,
    Conv2D,
    BatchNormalization,
    Activation,
    GlobalAveragePooling2D,
    AveragePooling2D,
)
from keras.models import Model


def basic_conv_block(input, chs, rep):
    x = input
    for i in range(rep):
        x = Conv2D(chs, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x


def create_cnn():
    input = Input(shape=(32, 32, 3))
    x = basic_conv_block(input, 64, 3)
    x = AveragePooling2D(2)(x)
    x = basic_conv_block(x, 128, 3)
    x = AveragePooling2D(2)(x)
    x = basic_conv_block(x, 256, 3)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation="softmax")(x)

    model = Model(input, x)
    return model
