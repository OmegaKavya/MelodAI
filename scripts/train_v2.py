#!/usr/bin/env python3
"""
Retrain with the data and augmentation problems fixed.

Three changes from scripts/train_segmented_model.py, each for a measured
reason rather than a hunch.

1. Songs are grouped. The old split was alphabetical over filenames; this one
   partitions by song, stratified per class, so no song's variants straddle
   the boundary and the holdout is a real holdout.

2. Contradictory labels are dropped. Five songs exist as byte-identical
   images under two classes, every one of them involving `peaceful`, the
   class that scored 9% per song.

3. Augmentation respects what the image is. The old pipeline applied
   photographic transforms to a spectrogram:

       rotation_range=15      rotates the time/frequency plane into itself
       shear_range=0.1        skews time against frequency
       height_shift_range     shifts the FREQUENCY axis, not time as labelled
       horizontal_flip=True   plays the song backwards
       brightness_range       rescales dB, undoing percentile normalisation

   None of those correspond to anything that can happen to audio. Rotating a
   spectrogram by 15 degrees makes a 200 Hz partial drift into a different
   band as time passes, which is not a sound. The green channel carries
   chroma, so a vertical shift transposes the key by a random amount while
   keeping the label.

   This script uses SpecAugment instead: masking bands of time and frequency,
   which corresponds to a dropout or an occlusion and leaves every surviving
   pixel meaning what it meant. Pitch and tempo augmentation already happen
   correctly, in the audio domain, when the spectrograms are generated.

Usage:
    python scripts/train_v2.py --epochs 60
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import CLASSES, scan, split_by_song, summarise  # noqa: E402

AUTOTUNE = tf.data.AUTOTUNE
IMG = 224


def spec_augment(image, max_time=28, max_freq=24, count=2):
    """Mask random bands of time and frequency.

    The standard augmentation for spectrogram models. Every pixel that
    survives still means what it meant, which is the property the geometric
    transforms destroyed.
    """
    for _ in range(count):
        t = tf.random.uniform([], 0, max_time, dtype=tf.int32)
        t0 = tf.random.uniform([], 0, IMG - t, dtype=tf.int32)
        mask = tf.concat([
            tf.ones([t0, IMG, 3]), tf.zeros([t, IMG, 3]), tf.ones([IMG - t0 - t, IMG, 3]),
        ], axis=0)
        image *= mask

        f = tf.random.uniform([], 0, max_freq, dtype=tf.int32)
        f0 = tf.random.uniform([], 0, IMG - f, dtype=tf.int32)
        mask = tf.concat([
            tf.ones([IMG, f0, 3]), tf.zeros([IMG, f, 3]), tf.ones([IMG, IMG - f0 - f, 3]),
        ], axis=1)
        image *= mask
    return image


def load(path, label, training):
    image = tf.io.decode_png(tf.io.read_file(path), channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, [IMG, IMG, 3])
    if training:
        # A small shift along time only. Time is the vertical axis here,
        # because the images were written with origin="lower".
        shift = tf.random.uniform([], -12, 12, dtype=tf.int32)
        image = tf.roll(image, shift, axis=0)
        image = spec_augment(image)
    return image, tf.one_hot(label, len(CLASSES))


def pipeline(items, training, batch=16):
    paths = tf.constant([i["path"] for i in items])
    labels = tf.constant([i["label"] for i in items])
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(items), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, l: load(p, l, training), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)


def build_model(num_classes=len(CLASSES)):
    """The same shape of network as before, kept so the comparison is fair."""
    def block(x, filters, pool=True):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return layers.MaxPooling2D()(x) if pool else x

    inputs = layers.Input((IMG, IMG, 3))
    x = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    for filters in (128, 256, 512):
        x = block(x, filters)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def evaluate_by_song(model, items, batch=16):
    """Per song, averaging a song's variants, which is how the app predicts."""
    ds = pipeline(items, training=False, batch=batch)
    probabilities = model.predict(ds, verbose=0)

    by_song = {}
    for item, probs in zip(items, probabilities):
        key = (item["label"], item["song"])
        by_song.setdefault(key, []).append(probs)

    correct = 0
    per_class = {c: [0, 0] for c in CLASSES}
    confusion = np.zeros((len(CLASSES), len(CLASSES)), int)
    for (label, _), probs in by_song.items():
        predicted = int(np.mean(probs, axis=0).argmax())
        confusion[label][predicted] += 1
        per_class[CLASSES[label]][1] += 1
        if predicted == label:
            correct += 1
            per_class[CLASSES[label]][0] += 1
    return correct, len(by_song), per_class, confusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="mel_spectrograms_resized")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--out", default="models/best_model_v2.h5")
    args = parser.parse_args()

    items = scan(args.data)
    train, validation, dropped = split_by_song(items)
    print(f"total images: {len(items)}")
    print(summarise(train, validation, dropped), "\n")

    counts = np.bincount([i["label"] for i in train], minlength=len(CLASSES))
    weights = {i: len(train) / (len(CLASSES) * c) for i, c in enumerate(counts) if c}
    print("class weights:", {CLASSES[i]: round(w, 3) for i, w in weights.items()}, "\n")

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    print(f"parameters: {model.count_params():,}\n")

    model.fit(
        pipeline(train, True),
        validation_data=pipeline(validation, False),
        epochs=args.epochs,
        class_weight=weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping("val_accuracy", patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=6, min_lr=1e-6),
        ],
        verbose=2,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)

    correct, total, per_class, confusion = evaluate_by_song(model, validation)
    print(f"\nper song: {correct}/{total} = {correct/total:.1%}\n")
    for name, (right, n) in per_class.items():
        if n:
            print(f"  {name:10} {right:2d}/{n:2d} = {right/n:6.1%}")
    print("\n  confusion (rows true, columns predicted):")
    print("    " + "".join(f"{c[:9]:>11}" for c in CLASSES))
    for i, name in enumerate(CLASSES):
        print(f"    {name[:9]:<9}" + "".join(f"{v:>11}" for v in confusion[i]))


if __name__ == "__main__":
    main()
