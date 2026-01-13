# train.py
import argparse
import json
import os
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds


def build_datasets(img_size: int, batch_size: int, val_split: float, test_split: float, seed: int):
    """
    TFDS cats_vs_dogs returns ~23k images.
    We create a deterministic split from TFDS split="train":
      train/val/test = (1 - val_split - test_split) / val_split / test_split

    TFDS provides (image, label) with as_supervised=True
    Labels: 0=cat, 1=dog (TFDS convention).
    """
    if val_split <= 0 or test_split <= 0:
        raise ValueError("val_split and test_split must be > 0")
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0")

    ds_full = tfds.load(
        "cats_vs_dogs",
        split="train",
        as_supervised=True,
        shuffle_files=True,
    )

    n = tf.data.experimental.cardinality(ds_full).numpy()
    if n <= 0:
        raise RuntimeError("Could not determine dataset size (cardinality).")

    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    ds_train_raw = ds_full.take(n_train)
    ds_val_raw = ds_full.skip(n_train).take(n_val)
    ds_test_raw = ds_full.skip(n_train + n_val).take(n_test)

    def preprocess(image, label):
        image = tf.image.resize(image, (img_size, img_size))
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(label, tf.float32)
        return image, label

    # Simple augmentation (kept modest) - matches the notebook
    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
        ]
    )

    def preprocess_with_aug(image, label):
        image, label = preprocess(image, label)
        image = augment(image, training=True)
        return image, label

    def make_ds(ds_raw, training: bool):
        if training:
            ds = ds_raw.map(preprocess_with_aug, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.shuffle(2_000, seed=seed, reshuffle_each_iteration=True)
        else:
            ds = ds_raw.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    ds_train = make_ds(ds_train_raw, training=True)
    ds_val = make_ds(ds_val_raw, training=False)
    ds_test = make_ds(ds_test_raw, training=False)

    return ds_train, ds_val, ds_test, n_train, n_val, n_test


def build_model(img_size: int, lr: float, dropout: float):
    """
    MobileNetV2 transfer learning:
    - base pretrained on ImageNet
    - freeze base
    - classification head (sigmoid)
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


def _to_py(x):
    # JSON-safe scalar conversion
    try:
        return float(x)
    except Exception:
        return x


def main():
    parser = argparse.ArgumentParser(description="Train Cats vs Dogs model (TFDS) with MobileNetV2.")
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--val-split", type=float, default=0.10)
    parser.add_argument("--test-split", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default="models/cats_dogs.keras")
    args = parser.parse_args()

    # Reduce TF logging noise (CPU-only in Codespaces)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    tf.random.set_seed(args.seed)

    ds_train, ds_val, ds_test, n_train, n_val, n_test = build_datasets(
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )

    print(f"Sizes | train={n_train} val={n_val} test={n_test}")
    print(
        "Config | "
        f"img_size={args.img_size} batch_size={args.batch_size} epochs={args.epochs} "
        f"lr={args.lr} dropout={args.dropout} seed={args.seed}"
    )

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

    # Evaluate on test set (matches notebook intent)
    test_metrics = model.evaluate(ds_test, verbose=0, return_dict=True)
    print("Test metrics:", test_metrics)

    # Pick "best" epoch by val_auc when available (history still useful even if epochs=1)
    best_epoch = None
    best_history_metrics = {}
    if "val_auc" in history.history and len(history.history["val_auc"]) > 0:
        vals = history.history["val_auc"]
        best_epoch = int(max(range(len(vals)), key=lambda i: vals[i]))
        best_history_metrics = {k: _to_py(v[best_epoch]) for k, v in history.history.items()}
    else:
        best_history_metrics = {k: _to_py(v[-1]) for k, v in history.history.items()}

    # Save model + metadata
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    metadata = {
        "model_path": str(model_path),
        "img_size": int(args.img_size),
        "labels": {"0": "cat", "1": "dog"},
        "tf_version": str(tf.__version__),
        "seed": int(args.seed),
        "splits": {
            "train": float(1.0 - args.val_split - args.test_split),
            "val": float(args.val_split),
            "test": float(args.test_split),
        },
        "sizes": {"train": int(n_train), "val": int(n_val), "test": int(n_test)},
        "train_config": {
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "dropout": float(args.dropout),
            "early_stopping": {"monitor": "val_auc", "mode": "max", "patience": 2, "restore_best_weights": True},
            "augmentation": {"RandomFlip": "horizontal", "RandomRotation": 0.05},
        },
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "history_best_metrics": best_history_metrics,
        "test_metrics": {k: _to_py(v) for k, v in test_metrics.items()},
    }

    meta_path = model_path.parent / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved model to: {model_path}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
