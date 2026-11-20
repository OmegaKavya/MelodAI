"""The energy head, as arithmetic.

Four matrix multiplies over a 1024-dimensional YAMNet embedding. Kept
separate from the training code because this is the only part that has to
run on the server, and it depends on nothing but NumPy.
"""
import numpy as np


def forward(x, w):
    """BatchNorm (folded) -> relu -> relu -> softmax. Dropout is inference-inert."""
    x = np.asarray(x, dtype=np.float32)
    x = x * w["scale"] + w["shift"]
    x = np.maximum(x @ w["w1"] + w["b1"], 0.0)
    x = np.maximum(x @ w["w2"] + w["b2"], 0.0)
    x = x @ w["w3"] + w["b3"]
    x = x - x.max(axis=-1, keepdims=True)  # stabilised softmax
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)
