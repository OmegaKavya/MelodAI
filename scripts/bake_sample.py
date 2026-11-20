"""Precompute one analysis and ship it with the client.

Most visitors will not have a Hindi film song to hand, and the page is worth
nothing to them until something has been analysed. This runs the real
pipeline over one track and writes the result as JSON, so "try a sample"
renders the genuine output instantly, with no upload and no waiting for a
sleeping free instance to wake.

It is the real output, not a mock: the same function the API calls, over real
audio. The page labels it as a prepared example so nobody mistakes it for
their own upload.
"""
import base64
import json
import os
import sys

import librosa
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from melodai.runtime import Runtime, pretty_title  # noqa: E402
from server.main import (  # noqa: E402
    MAX_SECONDS, SPECTROGRAM_HOP, SPECTROGRAM_RATE, SPECTROGRAM_SECONDS,
    YAMNET_RATE, _spectrogram_payload,
)

# A party track: it is the case where the model is both correct and visibly
# interesting, with a wide energy swing and unambiguous AudioSet tags.
SOURCE = "data/party/Abhi_Toh_Party_Shuru_Hui_Hai_Full_Video_Song_Khoobsurat_Badshah_Sonam_Kapoor_Aastha.mp3"


def main():
    path = os.path.join(ROOT, SOURCE)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}")

    runtime = Runtime(ROOT)
    y, _ = librosa.load(path, sr=SPECTROGRAM_RATE, mono=True, duration=MAX_SECONDS)
    y16 = librosa.resample(y, orig_sr=SPECTROGRAM_RATE, target_sr=YAMNET_RATE)

    result = runtime.analyse(y16.astype(np.float32))
    result["spectrogram"] = _spectrogram_payload(y.astype(np.float32))
    result["duration"] = float(len(y) / SPECTROGRAM_RATE)
    result["spectrogram_seconds"] = min(SPECTROGRAM_SECONDS, result["duration"])
    result["title"] = pretty_title(os.path.basename(SOURCE))
    result["sample"] = True

    out = os.path.join(ROOT, "web", "src", "sample.json")
    with open(out, "w") as handle:
        json.dump(result, handle, separators=(",", ":"))
    print(f"{result['title']}: {result['energy']['label']} "
          f"({result['energy']['score']:.3f}) -> {os.path.getsize(out) / 1024:.0f} kB")


if __name__ == "__main__":
    main()
