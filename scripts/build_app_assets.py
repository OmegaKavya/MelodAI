#!/usr/bin/env python3
"""
Build everything the app needs at runtime, so it never trains or re-embeds
the corpus on a visitor's request.

Produces:
  models/energy_model.keras   binary arousal head, the one thing that works
  models/corpus.npz           one pooled embedding per song, with names,
                              labels and 2-D coordinates
  models/projector.joblib     a fitted UMAP, so an uploaded song can be
                              placed on the same map

UMAP rather than t-SNE for the map. t-SNE has no transform: it can only
position points it was fitted on, so a visitor's song could never be drawn
alongside the corpus. UMAP learns a mapping that new points can pass
through, which is the whole feature.

Usage:
    .venv-yamnet/bin/python scripts/build_app_assets.py
"""
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_dataset import SOURCE_TO_CLASS, digest  # noqa: E402
from train_yamnet import build, load_cache  # noqa: E402

SOURCES = ["calm", "drive", "party", "romantic", "sad"]


def main():
    import joblib
    import tensorflow as tf
    import umap

    cache, _ = load_cache()

    found, names = collections.defaultdict(set), {}
    for folder in SOURCES:
        directory = os.path.join("data", folder)
        for name in sorted(os.listdir(directory)):
            if name.lower().endswith(".mp3"):
                key = digest(os.path.join(directory, name))
                found[key].add(folder)
                names.setdefault(key, name)

    keys, moods, titles, energy = [], [], [], []
    for key, sources in sorted(found.items()):
        if key not in cache:
            continue
        classes = {SOURCE_TO_CLASS[s] for s in sources}
        # A song in two folders keeps its energy label if both agree on it,
        # which is what matters here even when the mood label is ambiguous.
        high = {"party"} & sources
        if len(classes) > 1 and not high:
            continue
        keys.append(key)
        moods.append(sorted(classes)[0] if len(classes) == 1 else "ambiguous")
        titles.append(names[key])
        energy.append(1 if high else 0)

    X = np.stack([cache[k].mean(axis=0) for k in keys])
    energy = np.array(energy)
    print(f"corpus: {len(X)} songs, {int(energy.sum())} high energy")

    # ---- the energy model, trained on frames ----
    frame_x = np.concatenate([cache[k] for k in keys])
    frame_y = np.concatenate([np.full(len(cache[k]), e) for k, e in zip(keys, energy)])
    counts = np.bincount(frame_y, minlength=2)
    weights = {i: len(frame_y) / (2 * c) for i, c in enumerate(counts) if c}

    tf.keras.utils.set_random_seed(42)
    model = build(frame_x.shape[1], 2)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    model.fit(frame_x, frame_y, epochs=40, batch_size=256,
              class_weight=weights, verbose=0)

    correct = sum(int(model.predict(cache[k], verbose=0).mean(0).argmax()) == e
                  for k, e in zip(keys, energy))
    print(f"energy model on its own training songs: {correct}/{len(keys)} = {correct/len(keys):.1%}")
    model.save("models/energy_model.keras")

    # ---- the map ----
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.25, random_state=42)
    coords = reducer.fit_transform(X)
    joblib.dump(reducer, "models/projector.joblib")

    np.savez_compressed(
        "models/corpus.npz",
        embeddings=X.astype(np.float32),
        coords=coords.astype(np.float32),
        moods=np.array(moods),
        titles=np.array(titles),
        energy=energy,
    )
    print("wrote models/energy_model.keras, models/corpus.npz, models/projector.joblib")


if __name__ == "__main__":
    main()
