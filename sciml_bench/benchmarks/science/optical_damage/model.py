
import tensorflow as tf

# Model 
def autoencoder(input_shape=(150, 150, 3), latent_dim=2000):
    input_layer = tf.keras.layers.Input(input_shape)
    x = input_layer

    x = tf.keras.layers.Conv2D(
        64, kernel_size=3, kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)

    x = tf.keras.layers.Conv2D(
        32, kernel_size=3, kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)

    x = tf.keras.layers.Conv2D(
        16, kernel_size=3, kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.MaxPool2D(2)(x)

    # Bottleneck
    n_channels = 16
    h, w = input_shape[:2]
    h, w = h // 2**2, w // 2**2

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(latent_dim)(x)
    x = tf.keras.layers.Dense(h * w * n_channels)(x)
    x = tf.keras.layers.Reshape((h, w, n_channels))(x)

    # x = tf.keras.layers.Conv2DTranspose(16, strides=2, kernel_size=(
    #     3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(
        16, kernel_size=3, kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2DTranspose(32, strides=2, kernel_size=(
        3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(
        32, kernel_size=3, kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2DTranspose(64, strides=2, kernel_size=(
        3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(
        64, kernel_size=3, kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    output = tf.keras.layers.Conv2D(
        1, kernel_size=1, activation='linear', padding='same')(x)

    return tf.keras.models.Model(input_layer, output)