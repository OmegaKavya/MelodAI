#!/usr/bin/env python3
"""
Project the YAMNet embeddings and measure how separable the moods are.

This is the figure that explains the project. Every model tried reaches
roughly two thirds and fails the same way: `party` is recognised perfectly
and the quieter moods are confused with each other. The question is whether
that is a modelling failure or a property of the corpus, and a projection of
the representation answers it directly. If the quiet moods overlap in the
embedding space, no classifier built on that space can separate them, and
the ceiling is the data rather than the architecture.

The picture alone would only be suggestive, so it ships with numbers that do
not depend on the projection:

  * leave-one-out 1-NN agreement in the full 1024-d space, per class. If a
    song's nearest neighbour usually shares its label, the class is a real
    region; if not, it is not.
  * silhouette score, also in the full space, for how tight and separated
    each class is.

Both are computed before any dimensionality reduction, so they measure the
representation rather than the drawing.

Usage:
    .venv-yamnet/bin/python scripts/embedding_map.py
"""
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_dataset import SOURCE_TO_CLASS, digest  # noqa: E402
from train_yamnet import load_cache  # noqa: E402

SOURCES = ["calm", "drive", "party", "romantic", "sad"]
OUT = "models/embedding_map.png"

# Chosen to differ in hue and in lightness, so the classes stay distinct for
# a colourblind reader and in greyscale.
COLOURS = {"energetic": "#ff2e88", "emotional": "#ffc247", "peaceful": "#22e5ff"}
MARKERS = {"energetic": "o", "emotional": "s", "peaceful": "^"}


def manifest():
    found = collections.defaultdict(set)
    names = {}
    for folder in SOURCES:
        directory = os.path.join("data", folder)
        for name in sorted(os.listdir(directory)):
            if name.lower().endswith(".mp3"):
                key = digest(os.path.join(directory, name))
                found[key].add(folder)
                names.setdefault(key, name)
    return found, names


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    cache, _ = load_cache()
    found, names = manifest()

    keys, labels, titles = [], [], []
    for key, sources in sorted(found.items()):
        if key not in cache:
            continue
        classes = {SOURCE_TO_CLASS[s] for s in sources}
        if len(classes) > 1:      # contradictory labels, excluded throughout
            continue
        keys.append(key)
        labels.append(classes.pop())
        titles.append(names[key])

    # One point per song: the mean of its frame embeddings. That is also how
    # the classifier votes, so the picture shows what the classifier sees.
    X = np.stack([cache[k].mean(axis=0) for k in keys])
    labels = np.array(labels)
    print(f"songs: {len(X)}   dimensions: {X.shape[1]}")
    print("per class: " + "  ".join(f"{c}={int((labels==c).sum())}" for c in COLOURS))

    # ---- separability, measured in the full space ----
    print("\nnearest neighbour agreement (leave one out, full 1024-d space)")
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    _, indices = nn.kneighbors(X)
    neighbour = labels[indices[:, 1]]
    overall = (neighbour == labels).mean()
    for cls in COLOURS:
        mask = labels == cls
        share = (neighbour[mask] == cls).mean()
        chance = mask.mean()
        print(f"  {cls:10} {share:5.1%}   chance {chance:5.1%}   lift {share/chance:4.2f}x")
    print(f"  {'overall':10} {overall:5.1%}")

    numeric = np.array([sorted(COLOURS).index(c) for c in labels])
    print(f"\nsilhouette score: {silhouette_score(X, numeric):.3f} "
          "(1 is tight and separated, 0 is overlapping)")

    # ---- the projection ----
    coords = TSNE(n_components=2, perplexity=12, init="pca",
                  random_state=42, max_iter=2000).fit_transform(X)

    plt.rcParams.update({
        "figure.facecolor": "#0d0d0f", "axes.facecolor": "#0d0d0f",
        "text.color": "#f2f2f2", "axes.labelcolor": "#f2f2f2",
        "xtick.color": "#8a8a8a", "ytick.color": "#8a8a8a",
        "font.size": 11,
    })
    fig, ax = plt.subplots(figsize=(10.6, 7.2))
    for cls in sorted(COLOURS):
        mask = labels == cls
        ax.scatter(coords[mask, 0], coords[mask, 1], s=74,
                   c=COLOURS[cls], marker=MARKERS[cls],
                   edgecolors="#0d0d0f", linewidths=0.9,
                   label=f"{cls}  (n={int(mask.sum())})", alpha=0.92)

    ax.set_title("What the model can and cannot separate",
                 fontsize=15, fontweight="bold", pad=30, loc="left")
    ax.text(0, 1.035,
            "YAMNet embeddings, one point per song, projected with t-SNE",
            transform=ax.transAxes, fontsize=10, color="#9a9a9a")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#2a2a2e")
    # Outside the axes. t-SNE puts the dense region somewhere different on
    # every run, so any in-plot corner eventually collides with either the
    # data or an annotation.
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                       frameon=True, facecolor="#17171a",
                       edgecolor="#2a2a2e", fontsize=10.5)
    for text in legend.get_texts():
        text.set_color("#f2f2f2")

    # Annotate the two regions.
    #
    # Anchored at each group's centroid with the offset given in points, not
    # in data units: t-SNE's scale and orientation differ between runs, so a
    # data-space offset put one label outside the axes and the other in the
    # middle of the wrong cluster.
    energetic_centre = coords[labels == "energetic"].mean(axis=0)
    quiet_centre = coords[(labels == "emotional") | (labels == "peaceful")].mean(axis=0)
    ax.annotate("energetic:\nits own region", xy=energetic_centre,
                xytext=(0, 34), textcoords="offset points",
                ha="center", va="bottom", fontsize=11, color="#ff2e88",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="#0d0d0f", ec="#ff2e88", lw=0.8))
    ax.annotate("emotional and peaceful:\nno boundary between them", xy=quiet_centre,
                xytext=(0, -42), textcoords="offset points",
                ha="center", va="top", fontsize=11, color="#d2d2d2",
                bbox=dict(boxstyle="round,pad=0.35", fc="#0d0d0f", ec="#3a3a40", lw=0.8))

    ax.text(0.0, -0.06,
            f"Nearest-neighbour agreement: energetic "
            f"{(neighbour[labels=='energetic']=='energetic').mean():.0%}, "
            f"emotional {(neighbour[labels=='emotional']=='emotional').mean():.0%}, "
            f"peaceful {(neighbour[labels=='peaceful']=='peaceful').mean():.0%}."
            "  Measured in the full 1024-d space, not in this projection.",
            transform=ax.transAxes, fontsize=9, color="#7a7a7a", va="top")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="#0d0d0f", bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
