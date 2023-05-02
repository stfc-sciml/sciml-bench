from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers


def unet(input_shape: Tuple[int, int, int]) -> tf.keras.Model:

    input_layer = layers.Input(input_shape)
    x = input_layer

    # Encoder
    x = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.ReLU()(x)
    skip1 = x
    x = layers.MaxPool2D(2)(x)

    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.ReLU()(x)
    skip2 = x
    x = layers.MaxPool2D(2)(x)

    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.ReLU()(x)
    skip3 = x
    x = layers.MaxPool2D(2)(x)

    # Bottleneck
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.ReLU()(x)

    # Decoder
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate(axis=-1)([x, skip3])
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.ReLU()(x)

    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate(axis=-1)([x, skip2])
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.ReLU()(x)

    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate(axis=-1)([x, skip1])
    x = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(x)

    model = tf.keras.Model(input_layer, x)
    return model
