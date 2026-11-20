"""Serving runtime. NumPy and a TFLite interpreter, nothing else.

This replaces the TensorFlow path the Streamlit app used. The arithmetic is
the same and was checked against it, not assumed:

  * YAMNet runs from the vendored .tflite build. Embeddings agree with the
    TF Hub SavedModel to ~1e-5 and the resulting energy scores to four
    decimals, measured over six tracks.
  * The energy head is folded to plain arrays by scripts/export_energy_head.py,
    which refuses to write unless it reproduces Keras to 1e-5.
  * Map placement interpolates between the five nearest corpus songs rather
    than calling umap.transform. On unseen tracks the two agree to 13% of the
    map's span, and no choice of k or weighting got below that, so this is a
    genuine difference in method and not a tuning shortfall: the point is
    placed among its neighbours rather than re-embedded. Reproducing held out
    corpus points, which is the easier task, the error is 5.4%.

    Depending on umap.transform at serving time is not an option anyway.
    umap-learn 0.5.6 calls check_array(force_all_finite=...), which
    scikit-learn removed in 1.6, so the saved projector raises a TypeError on
    any current scikit-learn. That bug is live in the Streamlit build.

The point of all three is the same: serving should not need a 600 MB
training framework to replay four matrix multiplies.
"""
import csv
import os
import re

import numpy as np

from .head import forward

SAMPLE_RATE = 16000

# AudioSet's top of tree labels. Every piece of music scores high on "Music",
# so reporting it tells a listener nothing. The interesting mass is below.
GENERIC = {"Music", "Speech", "Musical instrument", "Sound effect", "Silence",
           "Inside, small room", "Audio logo"}

NEIGHBOURS_FOR_PLACEMENT = 5

# YAMNet is run over the track in fixed length chunks rather than in one call.
# A single invoke on two minutes of audio allocates intermediate tensors for
# every frame at once and pushed the server past 700 MB; in chunks it holds
# flat. Fixed length also means the interpreter allocates once and is reused,
# instead of reallocating for every different track duration.
#
# The cost is the frames that would have straddled a boundary: YAMNet hops
# 0.48s, so each join loses about one frame out of thirty. Over a full track
# that is under 3% of windows, and they are dropped rather than fabricated.
CHUNK_SECONDS = 15

# Two guards, both calibrated against the corpus rather than guessed.
#
# Hindi film music cannot be detected as such, so the page does not pretend
# to. What YAMNet can say is whether this is music at all: real songs score
# 0.55 to 0.97 on AudioSet "Music", while noise, a pure tone and silence
# score 0.00 to 0.05. Anything under 0.20 sits in the gap and is refused.
#
# The second is softer. Corpus songs sit at 0.81 to 0.94 cosine from their
# nearest neighbour; the same non music clips reach only 0.45 to 0.55. Below
# 0.70 the upload is outside the distribution the model was fitted on, so the
# verdict is reported with a warning rather than withheld.
MUSIC_FLOOR = 0.20
FAMILIAR_FLOOR = 0.70

# The corpus is keyed by the filename each track was downloaded under, which
# carries a tail of upload boilerplate and cast lists. Cut at the first of
# these and the song title is what survives.
TITLE_NOISE = {
    "full", "video", "song", "songs", "official", "lyrical", "lyric", "lyrics",
    "hd", "audio", "movie", "with", "ft", "feat", "featuring", "vi", "new",
    "latest", "bollywood", "series", "tseries", "4k", "remix", "version",
    "title", "track", "theme", "reprise",
}
TITLE_WORDS = 4


def pretty_title(name):
    """A filename, read back as a song title."""
    stem = re.sub(r"\.(mp3|wav|m4a|flac)$", "", str(name), flags=re.I)
    parts = [p for p in re.split(r"[_\-\s]+", stem) if p]
    keep = []
    for part in parts:
        if re.sub(r"[^a-z]", "", part.lower()) in TITLE_NOISE:
            break
        keep.append(part)
    keep = keep or parts[:TITLE_WORDS]
    words = []
    for word in keep:
        # "Let_S_Nacho" is an apostrophe the filename lost. Put it back.
        if len(word) == 1 and word.isalpha() and words:
            words[-1] = words[-1] + "'" + word.lower()
        else:
            words.append(word.capitalize())
    return " ".join(words[:TITLE_WORDS])


class Runtime:
    """Loaded once at startup and reused for every request."""

    def __init__(self, root="."):
        from ai_edge_litert.interpreter import Interpreter

        self.interpreter = Interpreter(
            model_path=os.path.join(root, "models/yamnet.tflite"))
        self._input = self.interpreter.get_input_details()
        self._output = self.interpreter.get_output_details()
        self._allocated = None

        with open(os.path.join(root, "models/yamnet_class_map.csv")) as handle:
            self.class_names = [row["display_name"] for row in csv.DictReader(handle)]

        self.head = dict(np.load(os.path.join(root, "models/energy_head.npz")))

        corpus = np.load(os.path.join(root, "models/corpus.npz"), allow_pickle=False)
        self.corpus_embeddings = corpus["embeddings"].astype(np.float32)
        self.corpus_coords = corpus["coords"]
        self.corpus_titles = corpus["titles"]
        self.corpus_moods = corpus["moods"]
        self.corpus_energy = corpus["energy"]
        self._corpus_unit = self.corpus_embeddings / (
            np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True) + 1e-9)

    # ---- the pieces -------------------------------------------------------

    def _invoke(self, waveform):
        """One interpreter pass, resizing only when the length changes."""
        if self._allocated != len(waveform):
            self.interpreter.resize_tensor_input(
                self._input[0]["index"], [len(waveform)], strict=True)
            self.interpreter.allocate_tensors()
            self._allocated = len(waveform)
        self.interpreter.set_tensor(self._input[0]["index"], waveform)
        self.interpreter.invoke()
        # Copied, because the next invoke overwrites the interpreter's buffers.
        return (
            np.array(self.interpreter.get_tensor(self._output[0]["index"])),
            np.array(self.interpreter.get_tensor(self._output[1]["index"])),
        )

    def embed(self, waveform):
        """YAMNet frame scores and embeddings, taken a chunk at a time."""
        waveform = np.ascontiguousarray(waveform, dtype=np.float32)
        size = CHUNK_SECONDS * SAMPLE_RATE

        if len(waveform) <= size:
            return self._invoke(waveform)

        scores, embeddings = [], []
        for start in range(0, len(waveform), size):
            chunk = waveform[start : start + size]
            # A tail shorter than one YAMNet window produces no frames and
            # would only cost an interpreter reallocation.
            if len(chunk) < SAMPLE_RATE:
                break
            s, e = self._invoke(np.ascontiguousarray(chunk))
            scores.append(s)
            embeddings.append(e)
        return np.concatenate(scores), np.concatenate(embeddings)

    def energy(self, embeddings):
        """One verdict for the track, plus the per-window curve behind it.

        Averaged over windows rather than taken from the first: a track can
        open on an intro that represents nothing about the rest of it.
        """
        probabilities = forward(embeddings, self.head)[:, 1]
        mean = float(probabilities.mean())
        return {
            "label": "High energy" if mean >= 0.5 else "Not high energy",
            "score": mean,
            "curve": [float(p) for p in probabilities],
            "peak": float(probabilities.max()),
            "trough": float(probabilities.min()),
            "swing": float(probabilities.max() - probabilities.min()),
        }

    def hears(self, scores, limit=8):
        """The AudioSet classes YAMNet assigns, minus the uninformative ones."""
        mean = scores.mean(axis=0)
        out = []
        for index in mean.argsort()[::-1]:
            name = self.class_names[index]
            if name in GENERIC:
                continue
            out.append({"name": name, "score": float(mean[index])})
            if len(out) == limit:
                break
        return out

    def _similarity(self, embedding):
        """Cosine rather than euclidean: embedding magnitude tracks loudness,
        and two masterings of the same arrangement should still count close."""
        a = embedding / (np.linalg.norm(embedding) + 1e-9)
        return self._corpus_unit @ a

    def neighbours(self, embedding, limit=5):
        similarity = self._similarity(embedding)
        order = similarity.argsort()[::-1][:limit]
        return [
            {
                "title": pretty_title(self.corpus_titles[i]),
                "mood": str(self.corpus_moods[i]),
                "similarity": float(similarity[i]),
            }
            for i in order
        ]

    def place(self, embedding):
        """Interpolate onto the fitted map from the nearest corpus songs.

        Weighted linearly by cosine similarity. Sharper weightings were
        measured and were not better: k=5 with linear weights was the closest
        of every combination tried to what umap.transform returns.
        """
        similarity = self._similarity(embedding)
        index = similarity.argsort()[::-1][:NEIGHBOURS_FOR_PLACEMENT]
        weights = np.clip(similarity[index], 0, None)
        total = weights.sum()
        if total <= 0:
            return self.corpus_coords[index[0]]
        weights = weights / total
        return (self.corpus_coords[index] * weights[:, None]).sum(axis=0)

    def is_music(self, scores):
        """Mean AudioSet "Music" score, and whether it clears the floor."""
        index = self.class_names.index("Music")
        score = float(scores.mean(axis=0)[index])
        return score, score >= MUSIC_FLOOR

    def analyse(self, waveform):
        scores, embeddings = self.embed(waveform)
        pooled = embeddings.mean(axis=0)
        position = self.place(pooled)
        music_score, _ = self.is_music(scores)
        closest = float(self._similarity(pooled).max())
        return {
            "music": {"score": music_score, "floor": MUSIC_FLOOR},
            "familiar": {
                "closest": closest,
                "floor": FAMILIAR_FLOOR,
                "in_distribution": closest >= FAMILIAR_FLOOR,
            },
            "energy": self.energy(embeddings),
            "hears": self.hears(scores),
            "neighbours": self.neighbours(pooled),
            "position": [float(position[0]), float(position[1])],
            "windows": int(len(embeddings)),
        }

    def corpus(self):
        """The map the uploaded track is placed on, for the client to draw."""
        return [
            {
                "title": pretty_title(self.corpus_titles[i]),
                "mood": str(self.corpus_moods[i]),
                "x": float(self.corpus_coords[i][0]),
                "y": float(self.corpus_coords[i][1]),
                "energy": int(self.corpus_energy[i]),
            }
            for i in range(len(self.corpus_coords))
        ]
