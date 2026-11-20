"""HTTP API for MelodAI.

One analysis endpoint, one corpus endpoint. Everything heavy is loaded once
at startup and shared; a request does signal processing and four matrix
multiplies, nothing more.

The spectrogram leaves here as raw bytes rather than a rendered PNG. The
client draws it, which keeps matplotlib and OpenCV out of the server and lets
the picture respond to a cursor instead of sitting there as a flat image.
"""
import base64
import io
import os

import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from melodai.runtime import Runtime, pretty_title
from melodai.spectrogram import hybrid_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# YAMNet wants 16 kHz; the spectrogram was trained at 22.05 kHz. Both are
# taken from the same decode rather than decoding the file twice.
YAMNET_RATE = 16000
SPECTROGRAM_RATE = 22050

# Two minutes is past the point where more audio changes the verdict, and it
# bounds what one request can cost.
MAX_SECONDS = 120
MAX_BYTES = 25 * 1024 * 1024

# The spectrogram is sent at the size it will be drawn at. Full resolution is
# 1723 columns for 20 seconds, which no screen resolves and no one should pay
# to download.
SPECTROGRAM_WIDTH = 512
SPECTROGRAM_HEIGHT = 128

app = FastAPI(title="MelodAI", docs_url="/api/docs", openapi_url="/api/openapi.json")

origins = os.environ.get("MELODAI_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

runtime = None


@app.on_event("startup")
def load():
    global runtime
    runtime = Runtime(ROOT)


def _decode(data: bytes):
    """Decode once, resample twice. Raises on anything librosa cannot read."""
    try:
        y, sr = librosa.load(io.BytesIO(data), sr=SPECTROGRAM_RATE,
                             mono=True, duration=MAX_SECONDS)
    except Exception as error:  # noqa: BLE001 - surfaced to the client as 400
        raise HTTPException(
            status_code=400,
            detail="That file could not be decoded. MP3, WAV and FLAC work.",
        ) from error
    if len(y) < sr:
        raise HTTPException(status_code=400, detail="That clip is under a second long.")
    y16 = librosa.resample(y, orig_sr=sr, target_sr=YAMNET_RATE)
    return y.astype(np.float32), y16.astype(np.float32)


def _downsample(img):
    """Block mean down to display size. Averaging rather than sampling, so a
    narrow percussive stripe survives instead of falling between columns."""
    height, width, _ = img.shape
    w = min(SPECTROGRAM_WIDTH, width)
    h = min(SPECTROGRAM_HEIGHT, height)
    columns = np.array_split(np.arange(width), w)
    rows = np.array_split(np.arange(height), h)
    out = np.empty((h, w, 3), dtype=np.float32)
    reduced = np.stack([img[:, c, :].mean(axis=1) for c in columns], axis=1)
    for i, r in enumerate(rows):
        out[i] = reduced[r, :, :].mean(axis=0)
    return out


def _spectrogram_payload(y):
    img = hybrid_image(y, sr=SPECTROGRAM_RATE)
    small = _downsample(img)
    # Row 0 is the lowest frequency; flip so the client can blit it straight
    # into a canvas, where row 0 is the top.
    small = small[::-1]
    pixels = (np.clip(small, 0, 1) * 255).astype(np.uint8)
    return {
        "width": int(pixels.shape[1]),
        "height": int(pixels.shape[0]),
        "rgb": base64.b64encode(pixels.tobytes()).decode("ascii"),
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "ready": runtime is not None}


@app.get("/api/corpus")
def corpus():
    """The map, so the client can draw it before anyone uploads anything."""
    return {"songs": runtime.corpus()}


@app.post("/api/analyse")
async def analyse(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="That file was empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="That file is over 25 MB.")

    y, y16 = _decode(data)
    result = runtime.analyse(y16)
    result["spectrogram"] = _spectrogram_payload(y)
    result["duration"] = float(len(y) / SPECTROGRAM_RATE)
    # The corpus is titled by the same rule, so an uploaded file and the
    # songs it sits beside read alike.
    result["title"] = pretty_title(file.filename or "Your track")
    result["filename"] = file.filename
    return JSONResponse(result)


# The built client is served by this same process when it is present, which is
# how the deployed Space runs: one origin, so no CORS and no base URL to keep
# in step. In development Vite serves the client and proxies /api here, and
# this block simply does not apply.
DIST = os.path.join(ROOT, "web", "dist")

if os.path.isdir(DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST, "assets")), name="assets")

    @app.get("/favicon.svg", include_in_schema=False)
    def favicon():
        return FileResponse(os.path.join(DIST, "favicon.svg"))

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(DIST, "index.html"))
