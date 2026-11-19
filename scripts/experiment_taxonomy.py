#!/usr/bin/env python3
"""
Does the label taxonomy explain the failure?

Every model so far, from two unrelated architectures and two unrelated
representations, fails on `peaceful` and fails on the training set too. That
is not overfitting and not capacity. The remaining suspect is the mapping
from five recorded categories to three labels.

This evaluates several groupings on identical data. Three properties make
the comparison honest:

  * The holdout is chosen once, stratified over the five source categories,
    and every grouping is scored on the same songs. Re-splitting per
    grouping would confound the taxonomy with the luck of the draw.
  * Songs ambiguous *under the grouping being tested* are dropped. Which
    songs those are changes with the grouping: calm and drive sharing seven
    songs is harmless when both map to one label and a contradiction when
    they do not.
  * Accuracy is reported against the chance baseline for that grouping,
    because 68% over three classes and 64% over five are not comparable
    numbers.

Usage:
    .venv-yamnet/bin/python scripts/experiment_taxonomy.py
"""
import collections
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_dataset import digest  # noqa: E402
from train_yamnet import build, load_cache, wilson  # noqa: E402

SOURCES = ["calm", "drive", "party", "romantic", "sad"]

GROUPINGS = {
    "as shipped": {
        "calm": "peaceful", "drive": "peaceful", "party": "energetic",
        "romantic": "emotional", "sad": "emotional",
    },
    "drive with party": {
        "calm": "calm", "drive": "energetic", "party": "energetic",
        "romantic": "emotional", "sad": "emotional",
    },
    "all five": {s: s for s in SOURCES},
    "arousal only": {  # high energy against everything else
        "calm": "low", "drive": "low", "party": "high",
        "romantic": "low", "sad": "low",
    },
}


def manifest():
    """Distinct audio, with every source folder it appears in."""
    found = collections.defaultdict(set)
    for folder in SOURCES:
        directory = os.path.join("data", folder)
        for name in sorted(os.listdir(directory)):
            if name.lower().endswith(".mp3"):
                found[digest(os.path.join(directory, name))].add(folder)
    return found


def main():
    import tensorflow as tf

    cache, _ = load_cache()
    songs = {k: v for k, v in manifest().items() if k in cache}
    print(f"songs with embeddings: {len(songs)}\n")

    # One holdout, chosen over the source categories, reused everywhere.
    rng = random.Random(42)
    by_source = collections.defaultdict(list)
    for key, sources in songs.items():
        by_source[sorted(sources)[0]].append(key)
    holdout = set()
    for source, keys in by_source.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        holdout |= set(keys[: max(1, round(len(keys) * 0.2))])
    print(f"fixed holdout: {len(holdout)} songs, same for every grouping\n")

    rows = []
    for title, mapping in GROUPINGS.items():
        classes = sorted(set(mapping.values()))
        labelled, dropped = {}, 0
        for key, sources in songs.items():
            targets = {mapping[s] for s in sources}
            if len(targets) > 1:
                dropped += 1          # ambiguous under THIS grouping
                continue
            labelled[key] = classes.index(targets.pop())

        train = [k for k in labelled if k not in holdout]
        val = [k for k in labelled if k in holdout]
        if len(val) < 5:
            print(f"{title}: too few validation songs, skipping")
            continue

        x = np.concatenate([cache[k] for k in train])
        y = np.concatenate([np.full(len(cache[k]), labelled[k]) for k in train])
        xv = np.concatenate([cache[k] for k in val])
        yv = np.concatenate([np.full(len(cache[k]), labelled[k]) for k in val])

        counts = np.bincount(y, minlength=len(classes))
        weights = {i: len(y) / (len(classes) * c) for i, c in enumerate(counts) if c}

        tf.keras.utils.set_random_seed(42)
        model = build(x.shape[1], len(classes))
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=["accuracy"])
        model.fit(x, y, validation_data=(xv, yv), epochs=150, batch_size=256,
                  class_weight=weights, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      "val_accuracy", patience=30, restore_best_weights=True)])

        correct = 0
        per = collections.defaultdict(lambda: [0, 0])
        for key in val:
            predicted = int(model.predict(cache[key], verbose=0).mean(0).argmax())
            truth = labelled[key]
            per[classes[truth]][1] += 1
            if predicted == truth:
                correct += 1
                per[classes[truth]][0] += 1

        # Also on training songs: a class the model cannot fit even here is a
        # labelling problem, not a generalisation problem.
        train_correct = sum(
            int(model.predict(cache[k], verbose=0).mean(0).argmax()) == labelled[k]
            for k in train)

        chance = 1 / len(classes)
        accuracy = correct / len(val)
        lo, hi = wilson(correct, len(val))
        rows.append((title, len(classes), dropped, correct, len(val), accuracy,
                     chance, accuracy / chance, lo, hi,
                     train_correct / len(train), dict(per)))

        print(f"--- {title}  ({len(classes)} classes, {dropped} songs dropped as ambiguous)")
        print(f"    val  {correct}/{len(val)} = {accuracy:.1%}   chance {chance:.0%}   "
              f"lift {accuracy/chance:.1f}x   CI {lo:.0%}-{hi:.0%}")
        print(f"    train {train_correct}/{len(train)} = {train_correct/len(train):.1%}")
        for cls, (right, n) in sorted(per.items()):
            print(f"      {cls:10} {right:2d}/{n:2d} = {right/n:6.1%}")
        print()

    print("=" * 74)
    print(f"{'grouping':20} {'classes':>8} {'val':>10} {'chance':>8} {'lift':>7} {'train':>8}")
    for r in sorted(rows, key=lambda r: -r[7]):
        print(f"{r[0]:20} {r[1]:>8} {r[5]:>9.1%} {r[6]:>8.0%} {r[7]:>6.1f}x {r[10]:>7.1%}")


if __name__ == "__main__":
    main()
