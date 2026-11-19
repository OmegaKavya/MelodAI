"""
MelodAI: what a network can actually tell about a Bollywood track.

This app used to predict one of three moods. It no longer does, because the
measurement said it could not. Five architectures across two representations
all reached about two thirds and all failed the same way, and a projection of
the embedding space showed why: energetic songs occupy their own region and
the quieter moods form one cloud with no boundary between them. Mainstream
Bollywood sits in a narrow band of tempo and production, a party track leaves
that band and a romantic or sad one does not.

So the prediction here is the one the data supports, and everything else on
the page is the working behind it: what the network hears, how the track
moves through itself, which songs it sits beside, and where it lands among
them. A visitor who doubts the verdict can see what produced it.
"""
import io
import os
import tempfile

import numpy as np
import streamlit as st

st.set_page_config(page_title="MelodAI", page_icon="🎵", layout="wide")

ACCENT = "#ff2e88"
ACCENT_2 = "#8c3cff"
COOL = "#22e5ff"
WARM = "#ffc247"
SURFACE = "#0d0d0f"

st.markdown(f"""
<style>
.stApp {{ background: {SURFACE}; }}
.verdict {{
  border: 1px solid #2a2a2e; border-radius: 14px; padding: 1.1rem 1.3rem;
  background: linear-gradient(135deg, rgba(255,46,136,.10), rgba(140,60,255,.10));
}}
.verdict h2 {{ margin: 0; font-size: 1.9rem; }}
.verdict p {{ margin: .25rem 0 0; opacity: .72; font-size: .9rem; }}
.tag {{
  display:inline-block; padding:.2rem .6rem; margin:.15rem .25rem .15rem 0;
  border-radius:999px; font-size:.8rem; border:1px solid #33333a;
}}
small.note {{ opacity:.55; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading YAMNet and the reference corpus...")
def get_analyzer():
    from melodai.analysis import Analyzer
    return Analyzer()


@st.cache_data(show_spinner=False)
def load_audio_16k(data: bytes, suffix: str):
    """Decode to the 16 kHz mono YAMNet requires."""
    import librosa
    path = tempfile.mktemp(suffix=suffix)
    with open(path, "wb") as handle:
        handle.write(data)
    try:
        y, _ = librosa.load(path, sr=16000, mono=True, duration=120)
        return np.clip(y.astype(np.float32), -1.0, 1.0)
    finally:
        if os.path.exists(path):
            os.remove(path)


@st.cache_data(show_spinner=False)
def hybrid_png(data: bytes, suffix: str):
    """The project's own spectrogram encoding, as an image.

    Red is percussive energy, green harmonic content blended with chroma,
    blue the mel spectrogram with spectral contrast. Kept because it is the
    representation the earlier CNN was trained on and it is worth seeing,
    even though the prediction above no longer comes from it.
    """
    import librosa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from melodai.spectrogram import hybrid_image

    path = tempfile.mktemp(suffix=suffix)
    with open(path, "wb") as handle:
        handle.write(data)
    try:
        y, sr = librosa.load(path, sr=22050, mono=True, duration=45)
        image = hybrid_image(y, sr)
    finally:
        if os.path.exists(path):
            os.remove(path)

    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor=SURFACE)
    ax.imshow(np.clip(image, 0, 1), aspect="auto", origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return buffer.getvalue()


def energy_curve_png(curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seconds = np.arange(len(curve)) * 0.48
    fig, ax = plt.subplots(figsize=(10, 2.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(seconds, curve, color=ACCENT, linewidth=2)
    ax.fill_between(seconds, 0, curve, color=ACCENT, alpha=0.16)
    ax.axhline(0.5, color="#4a4a52", linewidth=1, linestyle="--")
    ax.set_ylim(0, 1); ax.set_xlim(0, seconds[-1] if len(seconds) else 1)
    ax.set_ylabel("energy", color="#9a9a9a", fontsize=9)
    ax.set_xlabel("seconds", color="#9a9a9a", fontsize=9)
    ax.tick_params(colors="#6a6a72", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2a2a2e")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return buffer.getvalue()


def map_png(analyzer, position):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"energetic": ACCENT, "emotional": WARM, "peaceful": COOL,
               "ambiguous": "#55555c"}
    fig, ax = plt.subplots(figsize=(7.4, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for mood, colour in colours.items():
        mask = analyzer.corpus_moods == mood
        if not mask.any():
            continue
        ax.scatter(analyzer.corpus_coords[mask, 0], analyzer.corpus_coords[mask, 1],
                   s=42, c=colour, alpha=0.65, edgecolors=SURFACE, linewidths=0.6,
                   label=f"{mood} ({int(mask.sum())})")
    ax.scatter([position[0]], [position[1]], s=330, marker="*", c="#ffffff",
               edgecolors=ACCENT, linewidths=2, zorder=5, label="your track")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#2a2a2e")
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                       frameon=True, facecolor="#17171a", edgecolor="#2a2a2e",
                       fontsize=9)
    for text in legend.get_texts():
        text.set_color("#f2f2f2")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return buffer.getvalue()


# ---------------------------------------------------------------------------

st.markdown("# 🎵 MelodAI")
st.markdown(
    "**What a neural network can actually tell about a Bollywood track**, "
    "and what it cannot. Upload a song and watch the working."
)

with st.sidebar:
    st.markdown("### What this predicts")
    st.markdown(
        "**Energy, and only energy.** On a held-out set that prediction is "
        "right 26 times out of 26, where always guessing the majority would "
        "score 21 and catch none of the five high-energy tracks."
    )
    st.markdown("### What it refuses to predict")
    st.markdown(
        "Mood beyond energy. Five architectures across two representations all "
        "reached about two thirds and all failed the same way. In the "
        "embedding space, energetic songs form their own region while "
        "romantic, sad and calm form one cloud: nearest-neighbour agreement "
        "is **85%** for energetic and **45%** for the quiet moods, against a "
        "36% base rate."
    )
    st.markdown(
        "Mainstream Bollywood sits in a narrow band of tempo and production. "
        "A party track leaves that band. A romantic one does not. Arousal is "
        "recoverable from this audio; valence, at 131 songs, is not."
    )
    st.markdown("### The corpus")
    st.markdown("131 songs, 90 seconds each. YAMNet embeddings, 1024 dimensions.")
    st.markdown("[Source and method](https://github.com/OmegaKavya/MelodAI)")

uploaded = st.file_uploader("Upload an MP3 or WAV", type=["mp3", "wav", "m4a"])

if not uploaded:
    st.info(
        "Upload a track to see its energy over time, the AudioSet classes the "
        "network detects in it, the songs it sits nearest in the reference "
        "corpus, and where it lands on the map of that corpus."
    )
    st.stop()

data = uploaded.getvalue()
suffix = os.path.splitext(uploaded.name)[1] or ".mp3"
analyzer = get_analyzer()

with st.spinner("Listening..."):
    waveform = load_audio_16k(data, suffix)
    result = analyzer.analyse(waveform)

st.audio(data)

energy = result["energy"]
left, right = st.columns([3, 2])
with left:
    st.markdown(
        f"<div class='verdict'><h2>{energy['label']}</h2>"
        f"<p>{energy['score']:.0%} averaged over {result['windows']} windows "
        f"of 0.96 seconds. Peak {energy['peak']:.0%}, trough {energy['trough']:.0%}.</p></div>",
        unsafe_allow_html=True)
with right:
    st.metric("Swing across the track", f"{energy['swing']:.2f}",
              help="How far energy moves between the quietest and loudest window. "
                   "Above about 0.5 means the track genuinely changes character.")

st.markdown("### Energy through the track")
st.image(energy_curve_png(energy["curve"]))
st.markdown(
    "<small class='note'>One prediction per 0.96 second window. The dashed line "
    "is the decision boundary: above it the window reads as high energy.</small>",
    unsafe_allow_html=True)

st.markdown("### What the network hears")
st.markdown(
    "<small class='note'>YAMNet's final layer scores 521 AudioSet classes. These "
    "are the highest after removing the ones every piece of music scores on, "
    "like 'Music' and 'Speech'.</small>", unsafe_allow_html=True)
tags = "".join(
    f"<span class='tag' style='border-color:{ACCENT};color:#f2f2f2'>{name} "
    f"<b style='color:{ACCENT}'>{value:.3f}</b></span>"
    for name, value in result["hears"])
st.markdown(tags, unsafe_allow_html=True)

st.markdown("### Closest songs in the corpus")
st.markdown(
    "<small class='note'>Cosine similarity in the 1024-dimensional embedding "
    "space, not a genre lookup. This is what 'sounds alike' means to the "
    "model.</small>", unsafe_allow_html=True)
for neighbour in result["neighbours"]:
    if neighbour["similarity"] > 0.999:
        st.markdown(
            f"- **{neighbour['title'][:70]}** `{neighbour['similarity']:.3f}` "
            f"<small class='note'>this track is itself in the reference corpus</small>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"- {neighbour['title'][:70]} `{neighbour['similarity']:.3f}` "
            f"<small class='note'>[{neighbour['mood']}]</small>",
            unsafe_allow_html=True)

st.markdown("### Where it lands")
st.image(map_png(analyzer, result["position"]))
st.markdown(
    "<small class='note'>The reference corpus projected with UMAP, your track "
    "marked with a star. UMAP rather than t-SNE because t-SNE cannot place a "
    "point it was not fitted on, which would make this panel impossible.</small>",
    unsafe_allow_html=True)

st.markdown("### The spectrogram this project was built on")
with st.spinner("Rendering..."):
    st.image(hybrid_png(data, suffix))
st.markdown(
    "<small class='note'>The project's own encoding: red is percussive energy, "
    "green is harmonic content blended with chroma, blue is the mel spectrogram "
    "with spectral contrast. An earlier CNN was trained on exactly these "
    "images. The prediction above no longer comes from it, but the "
    "representation is worth seeing.</small>", unsafe_allow_html=True)
