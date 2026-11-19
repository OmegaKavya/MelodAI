"""
The guard against the bug that made the live demo meaningless.

The training pipeline and the Streamlit app each built their own spectrogram,
and they drifted: different hop length, different features, greyscale instead
of colour, and a 128-row image zero-padded into a 224x224 frame so that 43%
of what the model saw was black. Nothing failed loudly. The app returned a
confident-looking percentage the whole time.

These tests fail if the two ever diverge again.

    python -m pytest tests/ -v
"""
import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from melodai.spectrogram import (  # noqa: E402
    SAMPLE_RATE, TARGET_SIZE, hybrid_image, segments, to_model_input,
)

SR = SAMPLE_RATE


def tone(seconds=45, percussive=False, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    if percussive:
        y = 0.4 * np.sin(2 * np.pi * 110 * t) + 0.3 * rng.normal(size=t.size)
        y *= 0.3 + 0.7 * (np.sin(2 * np.pi * 2.5 * t) > 0.8)
    else:
        y = sum(0.25 * np.sin(2 * np.pi * f * t) for f in (220, 330, 440))
    return y.astype(np.float32)


def test_matches_the_training_script_exactly():
    """The whole point. Not "close enough": the same array.

    If this fails, the model is being shown something it was not trained on,
    and every prediction the app makes is unreliable even though nothing
    raises.
    """
    from generate_smart_segments import create_hybrid_spectrogram

    y = tone(percussive=True)

    reference_png = tempfile.mktemp(suffix=".png")
    assert create_hybrid_spectrogram(y, SR, reference_png)
    reference = cv2.resize(cv2.imread(reference_png), TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    os.remove(reference_png)

    assert np.array_equal(reference, to_model_input(hybrid_image(y, SR)))


def test_model_input_is_the_shape_the_network_expects():
    image = to_model_input(hybrid_image(tone(), SR))
    assert image.shape == (224, 224, 3)
    assert 0.0 <= image.min() and image.max() <= 1.0


def test_no_black_padding():
    """The old path padded a 128-row image into 224 rows of zeros.

    43% of the frame was black on a 45-second clip, and 90% on a short one.
    """
    image = to_model_input(hybrid_image(tone(), SR))
    black = (image.sum(axis=2) == 0).mean()
    # A few genuinely silent bins normalise to zero on a pure tone, around
    # 1.4% here. The threshold is set well above that and far below the 43%
    # the padding produced, so it fails on the bug and not on quiet audio.
    assert black < 0.10, f"{black:.0%} of the frame is black"


def test_channels_carry_different_information():
    """Red is percussive, green harmonic, blue mel.

    The old path stacked one greyscale array three times, so every
    cross-channel feature the model was built around was identically zero.
    """
    image = to_model_input(hybrid_image(tone(percussive=True), SR))
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    assert not np.allclose(r, g)
    assert not np.allclose(g, b)


def test_segments_split_at_the_trained_length():
    assert len(segments(tone(seconds=90), SR)) == 2
    assert len(segments(tone(seconds=45), SR)) == 1


def test_short_audio_is_used_whole_rather_than_padded():
    """Padding a short clip with silence makes a picture of nothing."""
    chunks = segments(tone(seconds=10), SR)
    assert len(chunks) == 1
    assert len(chunks[0]) == int(10 * SR)


@pytest.mark.skipif(not os.path.exists("models/best_model.h5"), reason="model not present")
def test_the_model_accepts_what_this_module_produces():
    import tensorflow as tf

    model = tf.keras.models.load_model("models/best_model.h5", compile=False)
    batch = np.stack([to_model_input(hybrid_image(tone(), SR))])
    probabilities = model.predict(batch, verbose=0)[0]
    assert probabilities.shape == (3,)
    assert abs(probabilities.sum() - 1.0) < 1e-4
