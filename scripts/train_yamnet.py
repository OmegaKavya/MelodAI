#!/usr/bin/env python3
"""
Classify mood from YAMNet embeddings.

The model here is deliberately small. The representation is doing the work:
each 0.96 second frame is already a 1024-d point in a space learned from two
million AudioSet clips, so what remains is a decision boundary, not feature
learning. Fitting a few hundred thousand parameters on 131 songs is a
reasonable ask; fitting five million was not.

Two prediction granularities, and the difference matters:

  frame   one embedding, one guess. What the network is trained on.
  song    every frame of a track averaged into one verdict. What a listener
          actually gets, and the only number worth reporting.

Trained on frames and reported on songs, because a song is the unit a person
cares about and because frame accuracy flatters itself: 93 frames from one
track are 93 highly correlated samples, not 93 observations.

Usage:
    .venv-yamnet/bin/python scripts/train_yamnet.py
"""
import argparse
import collections
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_dataset import CLASSES, scan, split_by_song, summarise  # noqa: E402

CACHE = "models/yamnet_embeddings.npz"


def load_cache():
    """Unpack the flat store back into one array per song."""
    data = np.load(CACHE, allow_pickle=False)
    counts = data["counts"]
    stacked = data["stacked"]
    out, offset = {}, 0
    for key, count in zip(data["keys"], counts):
        out[str(key)] = stacked[offset:offset + count]
        offset += count
    return out, {str(k): int(v) for k, v in zip(data["keys"], data["labels"])}


def wilson(correct, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = correct / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - spread) / d, (centre + spread) / d


def frames_for(items, cache):
    x, y, owner = [], [], []
    for item in items:
        frames = cache.get(item["key"])
        if frames is None:
            continue
        x.append(frames)
        y.append(np.full(len(frames), item["label"]))
        owner += [item["key"]] * len(frames)
    return np.concatenate(x), np.concatenate(y), owner


def build(input_dim, num_classes, hidden=256, dropout=0.5):
    import tensorflow as tf
    from tensorflow.keras import layers
    return tf.keras.Sequential([
        layers.Input((input_dim,)),
        # Normalising the embedding space first: AudioSet features have very
        # unequal scales, and without this the first dense layer spends its
        # capacity learning that rather than the task.
        layers.BatchNormalization(),
        layers.Dense(hidden, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(hidden // 2, activation="relu"),
        layers.Dropout(dropout / 2),
        layers.Dense(num_classes, activation="softmax"),
    ])


def report(model, items, cache, title):
    correct = 0
    per_class = {c: [0, 0] for c in CLASSES}
    confusion = np.zeros((len(CLASSES), len(CLASSES)), int)
    frame_correct = frame_total = 0

    for item in items:
        frames = cache.get(item["key"])
        if frames is None:
            continue
        probabilities = model.predict(frames, verbose=0)
        frame_correct += int((probabilities.argmax(1) == item["label"]).sum())
        frame_total += len(frames)

        predicted = int(probabilities.mean(axis=0).argmax())
        confusion[item["label"]][predicted] += 1
        per_class[item["mood"]][1] += 1
        if predicted == item["label"]:
            correct += 1
            per_class[item["mood"]][0] += 1

    total = sum(n for _, n in per_class.values())
    lo, hi = wilson(correct, total)
    print(f"\n{title}")
    print(f"  per frame: {frame_correct}/{frame_total} = {frame_correct/frame_total:.1%}")
    print(f"  per song : {correct}/{total} = {correct/total:.1%}   95% CI {lo:.0%} to {hi:.0%}")
    for cls, (right, n) in per_class.items():
        if n:
            print(f"     {cls:10} {right:2d}/{n:2d} = {right/n:6.1%}")
    print("     confusion (rows true, columns predicted):")
    print("       " + "".join(f"{c[:9]:>11}" for c in CLASSES))
    for i, name in enumerate(CLASSES):
        print(f"       {name[:9]:<9}" + "".join(f"{v:>11}" for v in confusion[i]))
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--out", default="models/yamnet_head.keras")
    args = parser.parse_args()

    import tensorflow as tf

    cache, _ = load_cache()
    items, conflicted = scan("data")
    train_items, val_items = split_by_song(items)
    print(summarise(items, conflicted, train_items, val_items), "\n")

    x_train, y_train, _ = frames_for(train_items, cache)
    x_val, y_val, _ = frames_for(val_items, cache)
    print(f"frames: {len(x_train)} train, {len(x_val)} val, dim {x_train.shape[1]}\n")

    counts = np.bincount(y_train, minlength=len(CLASSES))
    weights = {i: len(y_train) / (len(CLASSES) * c) for i, c in enumerate(counts) if c}

    model = build(x_train.shape[1], len(CLASSES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    print(f"parameters: {model.count_params():,}\n")

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs, batch_size=256,
        class_weight=weights, verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping("val_accuracy", patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=8, min_lr=1e-6),
        ],
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    report(model, train_items, cache, "TRAIN")
    score = report(model, val_items, cache, "VALIDATION")

    # The number that gets quoted has to come from the saved file, not from
    # the object in memory. The EfficientNet run reported 68% and reloaded at
    # 60%, and that discrepancy was only found by checking.
    reloaded = tf.keras.models.load_model(args.out)
    again = report(reloaded, val_items, cache, "VALIDATION, after reload")
    print(f"\nreload fidelity: {score:.1%} in memory, {again:.1%} from disk, "
          f"{'consistent' if abs(score - again) < 1e-9 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
