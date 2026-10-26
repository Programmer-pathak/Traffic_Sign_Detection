import csv
import os

import numpy as np
import tensorflow as tf

CLASSES = 43
DEFAULT_IMAGE_SIZE = (64, 64)


def load_records(data_dir="data", split="Train"):
    """Read image paths and labels without loading the image pixels into RAM."""
    csv_path = os.path.join(data_dir, f"{split}.csv")
    split_dir = os.path.join(data_dir, split)
    records = []

    if os.path.isfile(csv_path):
        with open(csv_path, mode="r", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                records.append((
                    os.path.join(data_dir, row["Path"]),
                    int(row["ClassId"]),
                ))
    elif split == "Train" and os.path.isdir(split_dir):
        for class_id in range(CLASSES):
            class_dir = os.path.join(split_dir, str(class_id))
            if os.path.isdir(class_dir):
                records.extend(
                    (os.path.join(class_dir, name), class_id)
                    for name in os.listdir(class_dir)
                    if os.path.isfile(os.path.join(class_dir, name))
                )
    else:
        raise FileNotFoundError(
            f"{split} dataset not found: expected '{csv_path}' or '{split_dir}'"
        )

    if not records:
        raise ValueError(f"No image records found for {split} in '{data_dir}'")
    return records


def stratified_split(labels, validation_fraction=0.2, random_state=42):
    """Split indices per class so every represented class is validated."""
    labels = np.asarray(labels, dtype=np.int32)
    rng = np.random.default_rng(random_state)
    train_indices = []
    validation_indices = []

    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        validation_count = max(1, int(round(len(class_indices) * validation_fraction)))
        if validation_count >= len(class_indices):
            validation_count = len(class_indices) - 1
        if validation_count < 1:
            raise ValueError(f"Class {class_id} needs at least two images to split.")
        validation_indices.extend(class_indices[:validation_count])
        train_indices.extend(class_indices[validation_count:])

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return np.asarray(train_indices), np.asarray(validation_indices)


def make_dataset(records, image_size=DEFAULT_IMAGE_SIZE, batch_size=64,
                 training=False, seed=42, class_weights=None):
    """Build a bounded-memory image pipeline with consistent full-image resizing."""
    paths = [record[0] for record in records]
    labels = np.asarray([record[1] for record in records], dtype=np.int32)
    if class_weights is None:
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    else:
        sample_weights = np.asarray(
            [class_weights[int(label)] for label in labels], dtype=np.float32
        )
        dataset = tf.data.Dataset.from_tensor_slices(
            (paths, labels, sample_weights)
        )

    if training:
        dataset = dataset.shuffle(
            min(len(records), 10000), seed=seed, reshuffle_each_iteration=True
        )

    def decode_image(path, label, sample_weight=None):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, image_size, antialias=True)
        image = tf.cast(image, tf.float32) / 255.0
        if sample_weight is None:
            return image, label
        return image, label, sample_weight

    dataset = dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, labels
