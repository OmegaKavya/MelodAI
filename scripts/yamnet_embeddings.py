#!/usr/bin/env python3
"""
Extract YAMNet embeddings for every song, once, and cache them.

Why this instead of the spectrogram CNN.

Everything so far has asked a network to learn what music sounds like from
131 songs. YAMNet was trained on AudioSet, roughly two million clips, and
already represents audio in a 1024 dimensional space where similar sounds sit
near each other. Classifying from that space is fitting a few thousand
parameters rather than a few million, which is the right size of problem for
this much data.

It also removes the rendering pipeline from inference entirely. No hybrid
image, no matplotlib round trip, no 224x224 resize, and therefore no way for
training and serving to drift apart again.

YAMNet takes 16 kHz mono waveforms and returns one 1024-d embedding per 0.96
second frame. A 90 second clip gives about 93 frames. Embeddings are cached
to an .npz because extraction is the slow part and the classifier above it
will be trained many times.

Usage:
    .venv-yamnet/bin/python scripts/yamnet_embeddings.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_dataset import scan, summarise  # noqa: E402

SAMPLE_RATE = 16000  # YAMNet's required input rate
CACHE = "models/yamnet_embeddings.npz"


def load_audio(path, seconds=90):
    import librosa
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=seconds)
    # YAMNet expects float32 in [-1, 1]; librosa already returns that range,
    # but a clip that was normalised louder would clip the model's front end.
    return np.clip(y.astype(np.float32), -1.0, 1.0)


def main():
    import tensorflow as tf
    import tensorflow_hub as hub

    items, conflicted = scan("data")
    print(summarise(items, conflicted), "\n")

    print("loading YAMNet from tfhub...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")

    embeddings, labels, keys, names = [], [], [], []
    started = time.time()
    for index, item in enumerate(items, 1):
        try:
            waveform = load_audio(item["path"])
            # YAMNet returns (scores, embeddings, log_mel). The embeddings are
            # the penultimate layer, which is what transfers.
            _, frames, _ = model(waveform)
            embeddings.append(np.asarray(frames, dtype=np.float32))
            labels.append(item["label"])
            keys.append(item["key"])
            names.append(item["name"])
        except Exception as error:
            print(f"  skipped {item['name'][:50]}: {error}")
            continue
        if index % 20 == 0 or index == len(items):
            print(f"  {index}/{len(items)}  {time.time()-started:.0f}s")

    # Frame counts differ by a frame or two between clips, so the arrays are
    # stored individually rather than stacked into one ragged block.
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(
        CACHE,
        labels=np.array(labels),
        keys=np.array(keys),
        names=np.array(names),
        counts=np.array([e.shape[0] for e in embeddings]),
        stacked=np.concatenate(embeddings, axis=0),
    )
    total = sum(e.shape[0] for e in embeddings)
    print(f"\ncached {len(embeddings)} songs, {total} frames, "
          f"{os.path.getsize(CACHE)/1e6:.1f} MB -> {CACHE}")


if __name__ == "__main__":
    main()
