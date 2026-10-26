import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPool2D,
    RandomBrightness,
    RandomContrast,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
)
from tensorflow.keras.models import Model


def build_traffic_sign_model(input_shape=(64, 64, 3), num_classes=43):
    """Build a regularized CNN with training-only, label-safe augmentation."""
    inputs = Input(shape=input_shape, name="sign_input")
    x = RandomRotation(0.04, fill_mode="reflect", name="random_rotation")(inputs)
    x = RandomTranslation(0.08, 0.08, fill_mode="reflect", name="random_translation")(x)
    x = RandomZoom(0.1, fill_mode="reflect", name="random_zoom")(x)
    x = RandomContrast(0.15, name="random_contrast")(x)
    x = RandomBrightness(0.12, value_range=(0, 1), name="random_brightness")(x)

    for block, filters in enumerate((24, 48, 96), start=1):
        x = Conv2D(
            filters, 3, padding="same", use_bias=False,
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"conv{block}_1",
        )(x)
        x = BatchNormalization(name=f"bn{block}_1")(x)
        x = tf.keras.layers.Activation("relu", name=f"relu{block}_1")(x)
        x = Conv2D(
            filters, 3, padding="same", use_bias=False,
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"conv{block}_2",
        )(x)
        x = BatchNormalization(name=f"bn{block}_2")(x)
        x = tf.keras.layers.Activation("relu", name=f"relu{block}_2")(x)
        x = MaxPool2D(pool_size=2, name=f"pool{block}")(x)
        x = Dropout(0.15 if block < 3 else 0.2, name=f"drop{block}")(x)

    x = GlobalAveragePooling2D(name="global_average_pool")(x)
    x = Dense(
        256, activation="relu", kernel_regularizer=regularizers.l2(1e-4),
        name="dense_features",
    )(x)
    x = Dropout(0.4, name="classifier_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="traffic_sign_output")(x)
    return Model(inputs=inputs, outputs=outputs, name="GTSRB_Classifier")


if __name__ == "__main__":
    build_traffic_sign_model().summary()
