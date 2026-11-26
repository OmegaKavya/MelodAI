"""The formats the upload accepts are the formats the server can decode.

M4A and AAC were accepted and advertised for as long as the upload existed,
and neither ever decoded: an upload is decoded from memory, and librosa only
falls back to ffmpeg for a path. These build each allowed format in memory
and put it through the server's own decode.
"""
import io
import re

import numpy as np
import pytest
import soundfile as sf

from melodai.runtime import pretty_title
from server import main

# How to write each allowed format with libsndfile. An extension added to
# ALLOWED without an entry here fails below rather than going unproven.
WRITERS = {
    ".wav": ("WAV", "PCM_16"),
    ".flac": ("FLAC", "PCM_16"),
    ".ogg": ("OGG", "VORBIS"),
    ".mp3": ("MP3", "MPEG_LAYER_III"),
}


def encoded(suffix, seconds=2):
    fmt, subtype = WRITERS[suffix]
    t = np.arange(main.SPECTROGRAM_RATE * seconds) / main.SPECTROGRAM_RATE
    buffer = io.BytesIO()
    sf.write(buffer, (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
             main.SPECTROGRAM_RATE, format=fmt, subtype=subtype)
    return buffer.getvalue()


@pytest.mark.parametrize("suffix", sorted(main.ALLOWED))
def test_every_allowed_format_decodes(suffix):
    y, y16 = main._decode(encoded(suffix))
    assert abs(len(y) - 2 * main.SPECTROGRAM_RATE) < main.SPECTROGRAM_RATE // 10
    assert len(y16) > 0


def test_the_messages_name_exactly_the_allowed_formats():
    named = {"." + word.lower() for word in re.findall(r"[A-Z0-9]{2,}", main.FORMATS)}
    assert named == main.ALLOWED


def test_an_ogg_upload_is_titled_without_its_extension():
    assert pretty_title("Tum_Hi_Ho.ogg") == "Tum Hi Ho"
