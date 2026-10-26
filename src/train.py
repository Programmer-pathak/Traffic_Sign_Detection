import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from model import build_traffic_sign_model
from preprocess import (
    CLASSES,
    DEFAULT_IMAGE_SIZE,
    load_records,
    make_dataset,
    stratified_split,
)


def plot_and_save_history(history, save_path="training_history.png"):
    """Save training and validation loss/accuracy curves."""
    epochs = range(1, len(history.history["accuracy"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history.history["accuracy"], label="Train")
    axes[0].plot(epochs, history.history["val_accuracy"], label="Validation")
    axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].plot(epochs, history.history["loss"], label="Train")
    axes[1].plot(epochs, history.history["val_loss"], label="Validation")
    axes[1].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    for axis in axes:
        axis.legend()
        axis.grid(True)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200)
    plt.close(figure)
    print(f"Training curves saved to '{save_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Train a generalizing 64x64 GTSRB traffic-sign CNN."
    )
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs (early stopping applies)")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--data_dir", type=str, default="data", help="GTSRB data directory")
    parser.add_argument("--model_path", type=str, default="traffic_sign_model.keras", help="Best model output path")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch_size must be positive")
    tf.keras.utils.set_random_seed(42)

    print("Loading GTSRB image paths (pixels are decoded in batches)...")
    records = load_records(args.data_dir, "Train")
    labels = np.asarray([label for _, label in records], dtype=np.int32)
    if np.any(labels < 0) or np.any(labels >= CLASSES):
        raise ValueError(f"Class labels must be in [0, {CLASSES - 1}].")
    train_indices, val_indices = stratified_split(labels)
    train_records = [records[index] for index in train_indices]
    val_records = [records[index] for index in val_indices]
    train_labels = labels[train_indices]
    class_counts = np.bincount(train_labels, minlength=CLASSES)
    if np.any(class_counts == 0):
        missing = np.flatnonzero(class_counts == 0).tolist()
        raise ValueError(f"Training split is missing traffic-sign classes: {missing}")
    class_weights = {
        class_id: float(len(train_labels) / (CLASSES * class_counts[class_id]))
        for class_id in range(CLASSES)
    }
    train_dataset, train_labels = make_dataset(
        train_records, DEFAULT_IMAGE_SIZE, args.batch_size, training=True,
        class_weights=class_weights,
    )
    val_dataset, _ = make_dataset(
        val_records, DEFAULT_IMAGE_SIZE, args.batch_size
    )
    print(f"Training images: {len(train_records)} | Validation images: {len(val_records)}")

    model = build_traffic_sign_model(
        input_shape=(*DEFAULT_IMAGE_SIZE, 3), num_classes=CLASSES
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=3e-4, weight_decay=1e-4
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    model_directory = os.path.dirname(os.path.abspath(args.model_path))
    os.makedirs(model_directory, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            filepath=args.model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        ),
    ]

    print("\nTraining from scratch; validation is stratified by sign class...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    plot_path = os.path.join(model_directory, "training_history.png")
    plot_and_save_history(history, plot_path)
    print(f"Best model saved to: {args.model_path}")
    print("Training completed.")


if __name__ == "__main__":
    main()
