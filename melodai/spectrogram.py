"""
The spectrogram representation, in one place.

This is the file that did not exist, and its absence is the bug. The training
pipeline built a three-channel engineered image - percussive content in red,
harmonic plus chroma in green, mel plus spectral contrast in blue - saved it
through matplotlib and resized it to 224x224. The Streamlit app built a plain
mel spectrogram at a different hop length, stacked it into three identical
greyscale channels, and zero-padded it into the corner of a 224x224 frame,
leaving 43% of the image black.

The model was therefore asked, in production, to classify pictures unlike
anything in its training set, and its confidences collapsed towards a third
each. The reported 66% accuracy was measured correctly on correctly built
images; it never described the deployed app.

Both paths import this module now, so they cannot drift again.
"""
import os
import tempfile

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter

SAMPLE_RATE = 22050
SEGMENT_SECONDS = 45
N_MELS = 128
HOP_LENGTH = 256
N_FFT = 2048
FMIN = 20
FMAX = 8000
TARGET_SIZE = (224, 224)


def _norm(x):
    """Percentile normalisation, as training used.

    The 2nd and 98th percentiles rather than min and max, so one loud frame
    cannot flatten the rest of the image into the bottom of the range.
    """
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)


def hybrid_image(y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH):
    """The three-channel image the model was trained on.

    Red is percussive energy, green is harmonic content blended with chroma,
    blue is the mel spectrogram blended with spectral contrast. Each channel
    is percentile-normalised and lightly blurred, exactly as in
    scripts/generate_smart_segments.py.

    hop_length is a parameter only so the server can draw a coarser image than
    training used. Anything feeding the model must keep the default, because
    the model learned on that hop; a display that is downsampled to 512 columns
    before it is sent cannot show the difference and costs several times the
    memory to compute at full resolution.
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, hop_length=hop_length,
        n_fft=N_FFT, fmin=FMIN, fmax=FMAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)

    harmonic, percussive = librosa.effects.hpss(y)
    mel_harmonic_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=harmonic, sr=sr, n_mels=N_MELS, hop_length=hop_length),
        ref=np.max,
    )
    mel_percussive_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=percussive, sr=sr, n_mels=N_MELS, hop_length=hop_length),
        ref=np.max,
    )
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)

    # Every feature has its own frame count; the shortest wins so the stack
    # is rectangular.
    min_time = min(mel_db.shape[1], mel_harmonic_db.shape[1], chroma.shape[1], contrast.shape[1])
    mel_db = mel_db[:, :min_time]
    mel_harmonic_db = mel_harmonic_db[:, :min_time]
    mel_percussive_db = mel_percussive_db[:, :min_time]
    chroma = chroma[:, :min_time]
    contrast = contrast[:, :min_time]

    # Chroma has 12 rows and has to span 128 to sit alongside the mel bands.
    chroma_norm = _norm(chroma)
    chroma_expanded = np.zeros((N_MELS, min_time))
    for i in range(12):
        start, end = i * 10, min(i * 10 + 11, N_MELS)
        chroma_expanded[start:end, :] = chroma_norm[i : i + 1, :]

    red = _norm(mel_percussive_db)
    green = _norm(mel_harmonic_db) * 0.5 + chroma_expanded * 0.5
    blue = _norm(mel_db) * 0.6 + np.resize(_norm(contrast), (N_MELS, min_time)) * 0.4

    img = np.clip(np.stack([red, green, blue], axis=-1), 0, 1)
    for c in range(3):
        img[..., c] = gaussian_filter(img[..., c], sigma=0.5)
    return img


def to_model_input(img):
    """Take the hybrid image through the same file round trip training used.

    Training wrote each image with plt.imsave(origin="lower") and then resized
    the saved PNG with cv2. That flips the image vertically and quantises it
    to 8 bits, and the model learned on the result, so inference has to do it
    too rather than resizing the array in memory.
    """
    # cv2 and matplotlib are imported here rather than at module scope. They
    # are needed only for this training-fidelity round trip; the server draws
    # the spectrogram from the raw array and should not have to install them.
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = tempfile.mktemp(suffix=".png")
    try:
        plt.imsave(path, img, origin="lower", dpi=300)
        # cv2 reads BGR and writes BGR, so a read/resize/write round trip is
        # channel neutral; the array below is converted back to RGB once.
        saved = cv2.imread(path)
        resized = cv2.resize(saved, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    finally:
        if os.path.exists(path):
            os.remove(path)


def segments(y, sr=SAMPLE_RATE, seconds=SEGMENT_SECONDS):
    """Split audio into the segment length the model was trained on.

    Training cut 90-second clips into two 45-second segments. A clip shorter
    than one segment is used whole rather than padded with silence, which
    would be a picture of nothing in the part that is padded.
    """
    size = int(seconds * sr)
    if len(y) <= size:
        return [y]
    return [y[i : i + size] for i in range(0, len(y) - size + 1, size)]


def prepare(audio_path, sr=SAMPLE_RATE, max_segments=4):
    """Audio file to a batch of model inputs, one per segment."""
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    chunks = segments(y, sr)[:max_segments]
    return np.stack([to_model_input(hybrid_image(c, sr)) for c in chunks])


# Index order of the model's three outputs.
CLASSES = ["Emotional", "Energetic", "Peaceful"]


def predict(model, audio_path, max_segments=4):
    """Average the model over every segment of the track.

    One 45-second window can land on an intro and miss the song. Averaging
    the per-segment probabilities is test-time augmentation: it costs another
    forward pass and is strictly better than trusting whichever window
    happened to come first.
    """
    batch = prepare(audio_path, max_segments=max_segments)
    probabilities = model.predict(batch, verbose=0)
    mean = probabilities.mean(axis=0)
    return {
        "label": CLASSES[int(np.argmax(mean))],
        "confidence": float(np.max(mean)),
        "probabilities": {name: float(p) for name, p in zip(CLASSES, mean)},
        "segments": int(len(batch)),
        "per_segment": [CLASSES[int(np.argmax(p))] for p in probabilities],
    }
