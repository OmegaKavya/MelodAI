#!/usr/bin/env python3
"""
Transfer learning: fit a head on pretrained features instead of learning
everything from 844 images.

Why. The from-scratch network reaches 97% training accuracy against 63%
validation. That gap is not a regularisation problem to be tuned away; a
4.8M parameter network is being asked to learn what edges and textures are
from a few hundred pictures. A backbone that already knows reduces the task
to fitting a classifier head on good features.

The features here are ImageNet's, because what is on disk is spectrogram
images. That is a real mismatch - ImageNet knows cats, not chroma - but the
early layers encode oriented edges, repetition and texture gradients, which
is most of what distinguishes a percussive spectrogram from a sustained one.
It is a well-used baseline for audio-as-image problems.

A backbone pretrained on audio (YAMNet, PANNs) would almost certainly beat
it, since it has heard music rather than seen photographs. That path needs
the source MP3s rather than the rendered spectrograms, so it is a separate
piece of work.

Two stages, which matters:
  1. Backbone frozen, head trained. A randomly initialised head sends large
     gradients backwards, and unfreezing first would destroy the pretrained
     weights before they were ever used.
  2. Top of the backbone unfrozen at a much lower learning rate.

Usage:
    python scripts/train_transfer.py --backbone efficientnet --epochs 40
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import CLASSES, scan, split_by_song, summarise  # noqa: E402
from train_v2 import IMG, spec_augment  # noqa: E402

AUTOTUNE = tf.data.AUTOTUNE

BACKBONES = {
    "efficientnet": (tf.keras.applications.EfficientNetB0,
                     tf.keras.applications.efficientnet.preprocess_input),
    "mobilenet": (tf.keras.applications.MobileNetV2,
                  tf.keras.applications.mobilenet_v2.preprocess_input),
    "resnet": (tf.keras.applications.ResNet50,
               tf.keras.applications.resnet.preprocess_input),
}


def make_pipeline(items, training, preprocess, batch=16):
    """Load, augment, then apply the backbone's own preprocessing.

    Order matters. SpecAugment masks to zero, which means silence in the
    normalised image space, so it has to happen before the backbone's
    rescaling shifts what zero means.
    """
    paths = tf.constant([i["path"] for i in items])
    labels = tf.constant([i["label"] for i in items])

    def load(path, label):
        image = tf.io.decode_png(tf.io.read_file(path), channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.ensure_shape(image, [IMG, IMG, 3])
        if training:
            image = tf.roll(image, tf.random.uniform([], -12, 12, tf.int32), axis=0)
            image = spec_augment(image)
        return preprocess(image * 255.0), tf.one_hot(label, len(CLASSES))

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(items), seed=42, reshuffle_each_iteration=True)
    return ds.map(load, num_parallel_calls=AUTOTUNE).batch(batch).prefetch(AUTOTUNE)


def build(backbone_fn, dropout=0.4):
    base = backbone_fn(include_top=False, weights="imagenet",
                       input_shape=(IMG, IMG, 3), pooling="avg")
    base.trainable = False
    inputs = layers.Input((IMG, IMG, 3))
    x = base(inputs, training=False)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout / 2)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def report(model, validation, preprocess, title):
    # Not train_v2.evaluate_by_song: that builds its own pipeline without the
    # backbone's preprocessing, which would score the model on inputs it was
    # never given during training.
    ds = make_pipeline(validation, False, preprocess)
    probabilities = model.predict(ds, verbose=0)
    by_song = {}
    for item, probs in zip(validation, probabilities):
        by_song.setdefault((item["label"], item["song"]), []).append(probs)

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

    print(f"\n{title}: {correct}/{len(by_song)} = {correct/len(by_song):.1%}")
    for cls, (right, n) in per_class.items():
        if n:
            print(f"   {cls:10} {right:2d}/{n:2d} = {right/n:6.1%}")
    print("   confusion (rows true, columns predicted):")
    print("     " + "".join(f"{c[:9]:>11}" for c in CLASSES))
    for i, name in enumerate(CLASSES):
        print(f"     {name[:9]:<9}" + "".join(f"{v:>11}" for v in confusion[i]))
    return correct / len(by_song)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="mel_spectrograms_resized")
    parser.add_argument("--backbone", default="efficientnet", choices=list(BACKBONES))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--finetune-epochs", type=int, default=25)
    parser.add_argument("--out", default="models/best_model_transfer.h5")
    args = parser.parse_args()

    backbone_fn, preprocess = BACKBONES[args.backbone]

    items = scan(args.data)
    train, validation, dropped = split_by_song(items)
    print(f"total images: {len(items)}")
    print(summarise(train, validation, dropped), "\n")

    counts = np.bincount([i["label"] for i in train], minlength=len(CLASSES))
    weights = {i: len(train) / (len(CLASSES) * c) for i, c in enumerate(counts) if c}

    train_ds = make_pipeline(train, True, preprocess)
    val_ds = make_pipeline(validation, False, preprocess)

    model, base = build(backbone_fn)
    print(f"backbone: {args.backbone}  total params: {model.count_params():,}  "
          f"trainable: {sum(np.prod(w.shape) for w in model.trainable_weights):,}\n")

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=["accuracy"])
    print("--- stage 1: frozen backbone ---")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              class_weight=weights, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  "val_accuracy", patience=10, restore_best_weights=True)])
    report(model, validation, preprocess, "after stage 1, per song")

    # Unfreeze the top third. Early layers hold generic edge and texture
    # filters that 844 images cannot improve on; the later ones are the
    # ImageNet-specific part worth adapting.
    base.trainable = True
    cut = int(len(base.layers) * 0.67)
    for layer in base.layers[:cut]:
        layer.trainable = False
    for layer in base.layers:
        # BatchNorm statistics must stay frozen: batches of 16 produce noisy
        # estimates that undo the pretrained normalisation.
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=["accuracy"])
    print(f"\n--- stage 2: fine-tuning from layer {cut} of {len(base.layers)} ---")
    print(f"trainable params: {sum(np.prod(w.shape) for w in model.trainable_weights):,}\n")
    model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs,
              class_weight=weights, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  "val_accuracy", patience=8, restore_best_weights=True)])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    report(model, validation, preprocess, "after fine-tuning, per song")


if __name__ == "__main__":
    main()
