"""Export the energy head as plain arrays, so serving needs no TensorFlow.

The head is BatchNorm -> Dense(256, relu) -> Dense(128, relu) -> Dense(2,
softmax): 299k parameters and four matrix multiplies. TensorFlow was in the
serving stack only to replay that arithmetic, and it brought numpy<2,
protobuf<5 and a Python ceiling with it.

The fold here is exact, not an approximation, and the script refuses to write
unless the NumPy forward pass reproduces Keras to within 1e-5 on the whole
corpus.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from melodai.head import forward  # noqa: E402


def main():
    import tensorflow as tf

    model = tf.keras.models.load_model(os.path.join(ROOT, "models/energy_model.keras"))
    bn, d1, _, d2, _, d3 = model.layers

    gamma, beta, mean, var = [np.asarray(w) for w in bn.get_weights()]
    epsilon = bn.get_config()["epsilon"]

    # BatchNorm at inference is an affine map. Fold it to a scale and shift
    # so serving does one multiply instead of carrying four vectors.
    scale = gamma / np.sqrt(var + epsilon)
    shift = beta - mean * scale

    weights = {
        "scale": scale.astype(np.float32),
        "shift": shift.astype(np.float32),
    }
    for i, layer in enumerate((d1, d2, d3), start=1):
        w, b = layer.get_weights()
        weights[f"w{i}"] = np.asarray(w, dtype=np.float32)
        weights[f"b{i}"] = np.asarray(b, dtype=np.float32)

    corpus = np.load(os.path.join(ROOT, "models/corpus.npz"), allow_pickle=False)
    sample = corpus["embeddings"].astype(np.float32)

    expected = model.predict(sample, verbose=0)
    got = forward(sample, weights)
    delta = float(np.abs(expected - got).max())
    print(f"checked {len(sample)} embeddings, max |keras - numpy| = {delta:.3e}")
    if delta > 1e-5:
        raise SystemExit(f"forward pass does not match Keras (delta {delta})")

    out = os.path.join(ROOT, "models/energy_head.npz")
    np.savez(out, **weights)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
