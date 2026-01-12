# train.py
import argparse
import json
import os
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds


def build_datasets(img_size: int, batch_size: int, val_split: float):
    """
    TFDS cats_vs_dogs returns ~23k images. We'll do a deterministic split.
    We use as_supervised=True to get (image, label) pairs.
    Labels: 0=cat, 1=dog (TFDS convention).
    """
    ds_train_full = tfds.load(
        "cats_vs_dogs",
        split="train",
        as_supervised=True,
        shuffle_files=True,
    )

    # Cache the cardinality for a deterministic split
    n = tf.data.experimental.cardinality(ds_train_full).numpy()
    if n <= 0:
        raise RuntimeError("Could not determine dataset size (cardinality).")

    n_val = int(n * val_split)
    n_train = n - n_val

    # Deterministic split: first n_train for train, last n_val for validation
    ds_train = ds_train_full.take(n_train)
    ds_val = ds_train_full.skip(n_train)

    def preprocess(image, label):
        image = tf.image.resize(image, (img_size, img_size))
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(label, tf.float32)
        return image, label

    num_calls = 2

    ds_train = (
        ds_train
        .map(preprocess, num_parallel_calls=num_calls)
        .shuffle(2_000)
        .batch(batch_size)
        .prefetch(1)
    )

    ds_val = (
        ds_val
        .map(preprocess, num_parallel_calls=num_calls)
        .batch(batch_size)
        .prefetch(1)
    )

    return ds_train, ds_val, n_train, n_val


def build_model(img_size: int, lr: float, dropout: float):
    """
    MobileNetV2 transfer learning:
    - base pretrained on ImageNet
    - freeze base
    - small classification head (sigmoid)
    """
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))

    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="Train Cats vs Dogs model (TFDS) with MobileNetV2.")
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--model-path", type=str, default="models/cats_dogs.keras")
    args = parser.parse_args()

    # Force CPU + reduce TF logging noise
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    ds_train, ds_val, n_train, n_val = build_datasets(
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )

    print(f"Train size: {n_train} | Val size: {n_val}")
    print(f"img_size={args.img_size} batch_size={args.batch_size} epochs={args.epochs}")

    model = build_model(img_size=args.img_size, lr=args.lr, dropout=args.dropout)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=2,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save model + metadata
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    metadata = {
    "model_path": str(model_path),
    "img_size": int(args.img_size),
    "labels": {"0": "cat", "1": "dog"},
    "tf_version": str(tf.__version__),
    "train_size": int(n_train),
    "val_size": int(n_val),
    "final_metrics": {str(k): float(v[-1]) for k, v in history.history.items()},
    }

    meta_path = model_path.parent / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved model to: {model_path}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
