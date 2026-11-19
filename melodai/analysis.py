"""
Everything the app computes for an uploaded track.

Deliberately more than a label. The classification this corpus supports is
one bit, high energy against the rest, and a page that prints one bit is
not worth a visit. What is worth a visit is showing the working: what the
network hears, how the track moves through its own 90 seconds, which songs
it sits near, and where it lands among them.

None of that is decoration. Each panel is a different view of the same 1024
dimensional embedding the prediction is made from, so a visitor who
disagrees with the verdict can see exactly what produced it.
"""
import csv
import os

import numpy as np

SAMPLE_RATE = 16000

# AudioSet's top of tree labels. Every piece of music scores high on "Music",
# so reporting it tells a listener nothing. The interesting mass is below.
GENERIC = {"Music", "Speech", "Musical instrument", "Sound effect", "Silence",
           "Inside, small room", "Audio logo"}


class Analyzer:
    """Holds the loaded models and the corpus. Built once, reused per request."""

    def __init__(self, root="."):
        import joblib
        import tensorflow as tf
        import tensorflow_hub as hub

        self.yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
        with open(self.yamnet.class_map_path().numpy().decode()) as handle:
            self.class_names = [row["display_name"] for row in csv.DictReader(handle)]

        self.energy_model = tf.keras.models.load_model(
            os.path.join(root, "models/energy_model.keras"))
        corpus = np.load(os.path.join(root, "models/corpus.npz"), allow_pickle=False)
        self.corpus_embeddings = corpus["embeddings"]
        self.corpus_coords = corpus["coords"]
        self.corpus_titles = corpus["titles"]
        self.corpus_moods = corpus["moods"]
        self.corpus_energy = corpus["energy"]
        self.projector = joblib.load(os.path.join(root, "models/projector.joblib"))

    # ---- the pieces -------------------------------------------------------

    def embed(self, waveform):
        """YAMNet frame scores and embeddings for one waveform."""
        scores, embeddings, _ = self.yamnet(waveform)
        return np.asarray(scores), np.asarray(embeddings)

    def energy(self, embeddings):
        """One verdict for the track, plus the per-window curve behind it.

        Averaged over windows rather than taken from the first: a track can
        open on an intro that represents nothing about the rest of it.
        """
        probabilities = self.energy_model.predict(embeddings, verbose=0)[:, 1]
        mean = float(probabilities.mean())
        return {
            "label": "High energy" if mean >= 0.5 else "Not high energy",
            "score": mean,
            "curve": probabilities,
            "peak": float(probabilities.max()),
            "trough": float(probabilities.min()),
            "swing": float(probabilities.max() - probabilities.min()),
        }

    def hears(self, scores, limit=8):
        """The AudioSet classes YAMNet assigns, minus the uninformative ones."""
        mean = scores.mean(axis=0)
        ranked = mean.argsort()[::-1]
        out = []
        for index in ranked:
            name = self.class_names[index]
            if name in GENERIC:
                continue
            out.append((name, float(mean[index])))
            if len(out) == limit:
                break
        return out

    def neighbours(self, embedding, limit=5):
        """Nearest songs in the corpus by cosine distance.

        Cosine rather than euclidean: embedding magnitude tracks loudness,
        and two arrangements of the same idea at different mastering levels
        should still count as close.
        """
        corpus = self.corpus_embeddings
        a = embedding / (np.linalg.norm(embedding) + 1e-9)
        b = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-9)
        similarity = b @ a
        order = similarity.argsort()[::-1][:limit]
        return [
            {
                "title": str(self.corpus_titles[i]),
                "mood": str(self.corpus_moods[i]),
                "similarity": float(similarity[i]),
            }
            for i in order
        ]

    def place(self, embedding):
        """Where this track falls on the corpus map."""
        return self.projector.transform(embedding.reshape(1, -1))[0]

    def analyse(self, waveform):
        scores, embeddings = self.embed(waveform)
        pooled = embeddings.mean(axis=0)
        return {
            "energy": self.energy(embeddings),
            "hears": self.hears(scores),
            "neighbours": self.neighbours(pooled),
            "position": self.place(pooled),
            "windows": int(len(embeddings)),
            "pooled": pooled,
        }
