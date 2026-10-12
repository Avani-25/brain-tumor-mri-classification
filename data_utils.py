"""Data loading, preprocessing (resize + normalize) and augmentation helpers."""
import tensorflow as tf
keras = tf.keras

from src.config import *

AUTOTUNE = tf.data.AUTOTUNE


def list_images(folder):
    """Return all image paths inside a folder."""
    return [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]


def build_augmentation():
    """Random augmentations applied to TRAINING images only (input range 0-255)."""
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.05, fill_mode="constant"),          # about +-18 degrees
            keras.layers.RandomZoom(0.15, fill_mode="constant"),              # zoom in/out up to 15%
            keras.layers.RandomTranslation(0.1, 0.1, fill_mode="constant"),   # shifts up to 10%
            keras.layers.RandomBrightness(0.2, value_range=(0, 255)),         # brightness change
        ],
        name="augmentation",
    )


def load_raw_dataset(directory, shuffle):
    """Load images from folders, resized to IMG_SIZE, one-hot labels, pixel values 0-255."""
    return keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=IMG_SIZE,      # resize to 224x224
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


def prepare(ds, augment=False):
    """Optionally augment, then normalize pixels to the 0-1 range."""
    rescale = keras.layers.Rescaling(1.0 / 255)
    if augment:
        aug = build_augmentation()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def get_datasets():
    """Return (train_ds, val_ds, test_ds) ready for model.fit / evaluate."""
    train_ds = prepare(load_raw_dataset(TRAIN_DIR, shuffle=True), augment=True)
    val_ds = prepare(load_raw_dataset(VALID_DIR, shuffle=False), augment=False)
    test_ds = prepare(load_raw_dataset(TEST_DIR, shuffle=False), augment=False)  # no shuffle, so labels stay aligned
    return train_ds, val_ds, test_ds


def get_class_weights():
    """Class weights = total / (n_classes * class_count), computed on the training set."""
    counts = [len(list_images(TRAIN_DIR / c)) for c in CLASS_NAMES]
    total = sum(counts)
    return {i: total / (NUM_CLASSES * n) for i, n in enumerate(counts)}