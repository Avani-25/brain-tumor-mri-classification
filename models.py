"""Model architectures."""
import numpy as np
import tensorflow as tf
keras = tf.keras

from src.config import *

layers = keras.layers


def build_custom_cnn(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """Custom CNN from scratch: 5 x (Conv -> BatchNorm -> ReLU -> MaxPool [-> Dropout]) + dense head."""
    inputs = keras.Input(shape=input_shape)
    x = inputs

    for filters, drop in [(32, 0.0), (64, 0.0), (128, 0.2), (256, 0.2), (256, 0.3)]:
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)          # stabilizes learning
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        if drop:
            x = layers.Dropout(drop)(x)             # reduces overfitting

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="custom_cnn")


# ----------------------------------------------------------------------------
# Transfer learning
# ----------------------------------------------------------------------------
BACKBONES = {
    "MobileNetV2": keras.applications.MobileNetV2,
    "ResNet50": keras.applications.ResNet50,
    "EfficientNetB0": keras.applications.EfficientNetB0,
}


def _add_preprocessing(name, x):
    """Convert our 0-1 input into the format each pretrained network expects.
    The conversion lives INSIDE the model, so the saved .h5 accepts the same 0-1 images as the custom CNN."""
    if name == "MobileNetV2":                      # expects [-1, 1]
        return layers.Rescaling(2.0, offset=-1.0, name="preprocess")(x)
    if name == "EfficientNetB0":                   # expects 0-255 (its own rescaling is inside the backbone)
        return layers.Rescaling(255.0, name="preprocess")(x)
    if name == "ResNet50":                         # expects 0-255, BGR order, ImageNet mean subtracted
        x = layers.Rescaling(255.0, name="to_255")(x)
        conv = layers.Conv2D(3, 1, name="preprocess", trainable=False)   # fixed 1x1 conv = RGB->BGR + mean shift
        x = conv(x)
        kernel = np.zeros((1, 1, 3, 3), dtype="float32")
        for c in range(3):
            kernel[0, 0, 2 - c, c] = 1.0
        bias = np.array([-103.939, -116.779, -123.68], dtype="float32")
        conv.set_weights([kernel, bias])
        return x
    raise ValueError(f"Unknown model: {name}")


def build_transfer_model(name, weights="imagenet", dropout=0.3):
    """Pretrained backbone (frozen) + new classification head for the 4 tumor classes."""
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = _add_preprocessing(name, inputs)

    base = BACKBONES[name](include_top=False, weights=weights, input_shape=(*IMG_SIZE, 3))
                                        
    base.trainable = False                         # phase 1: only the new head learns
    x = base(x, training=False)                    # keep BatchNorm in inference mode

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs, name=f"{name}_transfer")


def unfreeze_top_layers(model, n_layers=40):
    """Phase 2 (fine-tuning): unfreeze the last n layers of the backbone, but keep BatchNorm frozen."""
    base = next(l for l in model.layers if isinstance(l, keras.Model))
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return base



