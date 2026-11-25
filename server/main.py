"""HTTP API for MelodAI.

One analysis endpoint, one corpus endpoint. Everything heavy is loaded once
at startup and shared; a request does signal processing and four matrix
multiplies, nothing more.

The spectrogram leaves here as raw bytes rather than a rendered PNG. The
client draws it, which keeps matplotlib and OpenCV out of the server and lets
the picture respond to a cursor instead of sitting there as a flat image.
"""
import base64
import ctypes
import ctypes.util
import gc
import io
import os
import threading
import time
from collections import deque

import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
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

# Ninety seconds is past the point where more audio changes the verdict, and
# it is what the 512 MB host affords. Measured: the request plateaus near
# 440 MB here against 450 at two minutes, because the floor is librosa and
# numba rather than the audio, so there is nothing to win by cutting further.
MAX_SECONDS = 90
MAX_BYTES = 25 * 1024 * 1024
# Read granularity for the upload. Small enough that an oversized body is
# refused long before it can matter, large enough not to make syscalls the
# bottleneck.
CHUNK_BYTES = 64 * 1024

# One analysis is several seconds of CPU on a fractional instance, so an
# unthrottled endpoint is a denial of service with no exploit required. An
# in-memory window is enough here: this runs as a single process, and a limiter
# that forgets everything on restart is the correct trade for a free tier.
RATE_MAX = 10
RATE_WINDOW = 60.0

# Checked before decoding. The decoder would reject most of these anyway, but
# it reads the whole file first, and the message it gives is about codecs
# rather than about what this page wants.
ALLOWED = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}

# The spectrogram is sent at the size it will be drawn at. Full resolution is
# 1723 columns for 20 seconds, which no screen resolves and no one should pay
# to download.
SPECTROGRAM_WIDTH = 1024
SPECTROGRAM_HEIGHT = 128

# The spectrogram panel is drawn from one segment, not the whole upload.
# Harmonic/percussive separation over two minutes costs more memory than the
# rest of the request together, and 45 seconds is the segment length the
# convolutional model was trained on, so this is the picture it was shown.
SPECTROGRAM_SECONDS = 30

# Four times training's hop. The image is reduced to 512 columns before it is
# sent, so a finer hop would be averaged away on the way out, and computing it
# costs well over a hundred megabytes of peak memory. Anything that feeds the
# model still uses the training hop; this is display only.
SPECTROGRAM_HOP = 1024

app = FastAPI(title="MelodAI", docs_url="/api/docs", openapi_url="/api/openapi.json")

# The client is served from Firebase and the API from Render, so these are
# two origins and the browser will ask. Listed here rather than configured on
# the host, so a redeploy cannot lose them.
DEFAULT_ORIGINS = ",".join([
    "https://melodai.web.app",
    "https://melodai.firebaseapp.com",
    "http://localhost:5173",
    "http://localhost:5174",
])

origins = os.environ.get("MELODAI_ORIGINS", DEFAULT_ORIGINS).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

runtime = None


def _make_trim():
    """Return freed arena memory to the OS after a request, where we can.

    librosa and numba leave a few hundred megabytes of freed-but-retained
    arena behind. glibc only returns it on malloc_trim, so a long lived
    server otherwise sits at its high water mark forever, which is the
    difference between fitting a 512 MB host and being killed by it.

    Absent on macOS and musl, where this resolves to None and is skipped.
    """
    name = ctypes.util.find_library("c")
    if not name:
        return None
    try:
        libc = ctypes.CDLL(name)
        trim = libc.malloc_trim
    except (OSError, AttributeError):
        return None
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return trim


_trim = _make_trim()


def release_memory():
    gc.collect()
    if _trim is not None:
        _trim(0)


@app.on_event("startup")
def load():
    global runtime
    runtime = Runtime(ROOT)
    warm()


def warm():
    """Compile the numba paths before any request arrives.

    librosa JIT compiles on first call, which costs tens of seconds on a
    fractional CPU. Left to the first request, every cold start hands one
    visitor that bill; done here it happens while the host is still waiting
    for the port to open.
    """
    try:
        tone = np.sin(np.linspace(0, 220 * 2 * np.pi, SPECTROGRAM_RATE * 2)).astype(np.float32)
        hybrid_image(tone, sr=SPECTROGRAM_RATE, hop_length=SPECTROGRAM_HOP)
        runtime.analyse(librosa.resample(
            tone, orig_sr=SPECTROGRAM_RATE, target_sr=YAMNET_RATE))
    except Exception:  # noqa: BLE001 - warming is an optimisation, not a gate
        pass


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
    y = y[: SPECTROGRAM_SECONDS * SPECTROGRAM_RATE]
    img = hybrid_image(y, sr=SPECTROGRAM_RATE, hop_length=SPECTROGRAM_HOP)
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


_rate_hits = {}
_rate_lock = threading.Lock()
# Hard ceiling on tracked addresses. Pruning only drops entries older than the
# window, so a burst from more distinct addresses than this inside one window
# would otherwise grow the map without bound.
RATE_MAX_TRACKED = 4096

# One analysis at a time. FastAPI runs a sync endpoint on a 40-thread pool, and
# one analysis already plateaus near 440 MB on a 512 MB instance, so two
# overlapping uploads were enough to get the process killed for everyone.
# Taken before the upload is read, so a waiting request holds no audio in
# memory (the multipart body is spooled to disk by then).
_analysis_slot = threading.BoundedSemaphore(1)
ANALYSIS_WAIT_SECONDS = 30


def _rate_limit(request: Request):
    """Per-IP sliding window over the analyse endpoint.

    Deliberately not a library. slowapi would be another dependency on an
    install that was cut to 441 MB on purpose, and the whole limiter is
    fifteen lines. Keyed on the forwarded client address, since the service
    sits behind Render's proxy and request.client.host is the proxy.
    """
    # Take the RIGHTMOST entry, not the leftmost.
    #
    # X-Forwarded-For is a list the proxies append to, so the leftmost value is
    # whatever the caller sent and is trivially forged: a script sending a
    # random address per request evades the limit entirely, and one sending a
    # real address can exhaust someone else's allowance on purpose. The last
    # entry is the one Render's proxy added itself, which is the only part of
    # the header the client cannot write.
    #
    # If the service is ever put behind a second proxy, this needs to skip that
    # many entries from the right instead.
    forwarded = [part.strip() for part in
                 request.headers.get("x-forwarded-for", "").split(",") if part.strip()]
    client = forwarded[-1] if forwarded else (
        request.client.host if request.client else "unknown")

    now = time.monotonic()
    with _rate_lock:
        seen = _rate_hits.setdefault(client, deque())
        while seen and now - seen[0] > RATE_WINDOW:
            seen.popleft()
        if len(seen) >= RATE_MAX:
            retry = int(RATE_WINDOW - (now - seen[0])) + 1
            raise HTTPException(
                status_code=429,
                detail=f"That is {RATE_MAX} analyses in a minute. "
                       f"Try again in {retry}s.",
                headers={"Retry-After": str(retry)},
            )
        seen.append(now)

        # Without this the dict grows one entry per distinct address, forever.
        if len(_rate_hits) > 2048:
            for key in [k for k, v in _rate_hits.items() if not v or now - v[-1] > RATE_WINDOW]:
                _rate_hits.pop(key, None)
        if len(_rate_hits) > RATE_MAX_TRACKED:
            # Still over after pruning: evict the least recently seen.
            by_age = sorted(_rate_hits, key=lambda k: _rate_hits[k][-1] if _rate_hits[k] else 0)
            for key in by_age[:len(_rate_hits) - RATE_MAX_TRACKED]:
                if key != client:
                    _rate_hits.pop(key, None)


@app.post("/api/analyse")
def analyse(request: Request, file: UploadFile = File(...)):
    """Deliberately a sync def.

    Decoding, YAMNet and the spectrogram are blocking CPU work. Declared
    async, they run on the event loop and nothing else is served for the
    duration, including /api/health. On a host that health checks every few
    seconds and restarts on failure, that turns every analysis into a
    restart, which is exactly what it did. As a sync def, FastAPI runs this
    in a worker thread and the loop stays free to answer.
    """
    _rate_limit(request)

    # Checked before a byte of the body is read. It used to be checked after
    # buffering up to 25 MB, and skipped entirely when there was no extension,
    # so an extensionless upload went straight to the audio decoders.
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in ALLOWED:
        raise HTTPException(
            status_code=415,
            detail=f"{suffix or 'A file with no extension'} is not an audio "
                   "format this reads. MP3, WAV, FLAC, M4A, OGG and AAC all work.",
        )

    if not _analysis_slot.acquire(timeout=ANALYSIS_WAIT_SECONDS):
        raise HTTPException(
            status_code=503,
            detail="Someone else's song is being analysed. Try again in a "
                   "few seconds.",
            headers={"Retry-After": "10"},
        )
    try:
        return _analyse_upload(file)
    finally:
        _analysis_slot.release()


def _analyse_upload(file):
    # Read in chunks and stop at the cap rather than buffering the whole body
    # and measuring it afterwards. A single 1 GB upload against a 512 MB
    # instance is otherwise an out-of-memory kill, which takes the service down
    # for everyone and costs the attacker one request.
    chunks = []
    total = 0
    while True:
        chunk = file.file.read(CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail="That file is over the 25 MB limit, which is a full "
                       "length song at a normal bitrate.",
            )
        chunks.append(chunk)

    data = b"".join(chunks)
    del chunks
    if not data:
        raise HTTPException(status_code=400, detail="That file was empty.")

    y, y16 = _decode(data)
    duration = float(len(y) / SPECTROGRAM_RATE)

    result = runtime.analyse(y16)
    del y16

    # Refusing is better than printing a confident verdict about a podcast.
    if result["music"]["score"] < result["music"]["floor"]:
        raise HTTPException(
            status_code=422,
            detail="That does not sound like music. This reads songs: it was "
                   "fitted on Hindi film tracks, and speech or noise gives it "
                   "nothing to work with.",
        )
    result["spectrogram"] = _spectrogram_payload(y)
    del y

    result["duration"] = duration
    result["spectrogram_seconds"] = min(SPECTROGRAM_SECONDS, duration)
    # The corpus is titled by the same rule, so an uploaded file and the
    # songs it sits beside read alike.
    result["title"] = pretty_title(file.filename or "Your track")
    result["filename"] = file.filename

    release_memory()
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
