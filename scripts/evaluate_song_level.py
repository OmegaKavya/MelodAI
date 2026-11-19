#!/usr/bin/env python3
"""
Evaluate per song, not per image, and report an interval rather than a number.

Why this exists.

Training writes eight images per song: two 45-second segments times four
augmentations. Keras splits `flow_from_directory` alphabetically, and since
every variant shares the song's filename prefix, songs stay whole apart from
the one that straddles the boundary. Leakage is therefore small, measured at
one song and 1.7% of validation images.

The real problem is different. Those 236 validation images are not 236
independent observations; they are roughly 30 songs seen eight ways each.
Quoting accuracy over the image count implies a precision the data cannot
support:

    236 "samples"  ->  66%, 95% CI 60-72%   (what the number looks like)
     30 songs      ->  66%, 95% CI 48-80%   (what it actually is)

A single misread song moves the figure by about three points. So this script
aggregates every image belonging to a song into one prediction, the same way
the app averages a track's segments, and reports the Wilson interval on the
song count.

Usage:
    python scripts/evaluate_song_level.py --data ../mel_spectrograms_resized
"""
import argparse
import collections
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def song_of(filename):
    """Strip the segment and augmentation suffix to get the song.

    "{song}_{seg1|seg2}_{aug}.png" is the scheme generate_smart_segments.py
    writes. Splitting on "_seg" is exact for it and leaves any filename
    without that marker as its own song, which is the safe failure.
    """
    return os.path.basename(filename).split("_seg")[0]


def wilson(successes, n, z=1.96):
    """Wilson score interval: behaves at small n, where normal approximation does not."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - spread) / d, (centre + spread) / d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../mel_spectrograms_resized")
    parser.add_argument("--model", default="models/best_model.h5")
    parser.add_argument("--split", type=float, default=0.2)
    args = parser.parse_args()

    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    model = tf.keras.models.load_model(args.model, compile=False)

    # The same generator training used, so this evaluates the same holdout
    # rather than a differently drawn one.
    gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=args.split).flow_from_directory(
        args.data, target_size=(224, 224), batch_size=32,
        class_mode="categorical", subset="validation", shuffle=False, seed=42,
    )

    classes = [name for name, _ in sorted(gen.class_indices.items(), key=lambda kv: kv[1])]
    probabilities = model.predict(gen, verbose=0)
    truths = gen.classes

    # ---- per image, which is what the old figure reported -----------------
    image_correct = int((probabilities.argmax(axis=1) == truths).sum())
    image_total = len(truths)

    # ---- per song ---------------------------------------------------------
    by_song = collections.defaultdict(lambda: {"probs": [], "truth": None})
    for path, probs, truth in zip(gen.filenames, probabilities, truths):
        key = (truth, song_of(path))
        by_song[key]["probs"].append(probs)
        by_song[key]["truth"] = truth

    song_correct = 0
    per_class = collections.defaultdict(lambda: [0, 0])
    confusion = np.zeros((len(classes), len(classes)), dtype=int)

    for (truth, _), record in by_song.items():
        predicted = int(np.mean(record["probs"], axis=0).argmax())
        confusion[truth][predicted] += 1
        per_class[truth][1] += 1
        if predicted == truth:
            song_correct += 1
            per_class[truth][0] += 1

    song_total = len(by_song)

    print(f"\nValidation: {image_total} images, {song_total} songs\n")
    lo, hi = wilson(image_correct, image_total)
    print(f"  per image : {image_correct}/{image_total} = {image_correct/image_total:6.1%}"
          f"   95% CI {lo:.0%} to {hi:.0%}   <- overstates precision")
    lo, hi = wilson(song_correct, song_total)
    print(f"  per song  : {song_correct}/{song_total} = {song_correct/song_total:6.1%}"
          f"   95% CI {lo:.0%} to {hi:.0%}   <- the honest figure\n")

    print("  per class, by song:")
    for index, name in enumerate(classes):
        correct, total = per_class[index]
        if total:
            print(f"    {name:12} {correct:3d}/{total:3d} = {correct/total:5.1%}")

    print("\n  confusion (rows true, columns predicted):")
    print("    " + "".join(f"{c[:9]:>10}" for c in classes))
    for index, name in enumerate(classes):
        print(f"    {name[:9]:<9}" + "".join(f"{v:>10}" for v in confusion[index]))

    # ---- leakage check ----------------------------------------------------
    train_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=args.split).flow_from_directory(
        args.data, target_size=(224, 224), batch_size=32,
        class_mode="categorical", subset="training", shuffle=False, seed=42,
    )
    train_songs = {(t, song_of(f)) for f, t in zip(train_gen.filenames, train_gen.classes)}
    shared = {k for k in by_song if k in train_songs}
    leaked = sum(1 for f, t in zip(gen.filenames, truths) if (t, song_of(f)) in train_songs)

    print(f"\n  songs in both splits: {len(shared)}"
          f"  ({leaked}/{image_total} validation images, {leaked/image_total:.1%})")
    if shared:
        # Named with their class: the same song title can exist under two
        # moods, and printing the bare name made one leak look like three.
        named = sorted(f"{classes[t]}/{s}" for t, s in shared)
        print("  " + ", ".join(named)[:300])


if __name__ == "__main__":
    main()
