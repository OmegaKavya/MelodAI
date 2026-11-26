"""HTTP API for MelodAI.

One analysis endpoint, one corpus endpoint. Everything heavy is loaded once
at startup and shared; a request does signal processing and four matrix
multiplies, nothing more.

The spectrogram leaves here as raw bytes rather than a rendered PNG. The
client draws it, which keeps matplotlib and OpenCV out of the server and lets
the picture respond to a cursor instead of sitting there as a flat image.
"""
import asyncio
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
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header
from starlette.datastructures import UploadFile

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
# The request carries the file plus its multipart framing, the boundary and
# part headers, which come to a few hundred bytes. The body cap sits a little
# above the file cap so a file right at the limit still arrives whole.
MAX_BODY_BYTES = MAX_BYTES + 64 * 1024

# One analysis is several seconds of CPU on a fractional instance, so an
# unthrottled endpoint is a denial of service with no exploit required. An
# in-memory window is enough here: this runs as a single process, and a limiter
# that forgets everything on restart is the correct trade for a free tier.
RATE_MAX = 10
RATE_WINDOW = 60.0

# Checked before decoding. The decoder would reject the rest anyway, but it
# reads the whole file first, and the message it gives is about codecs rather
# than about what this page wants.
#
# Exactly what libsndfile decodes from memory. M4A and AAC are absent because
# they never could be: librosa only falls back to audioread, and so to ffmpeg,
# when given a path, and an upload is decoded from bytes.
ALLOWED = {".mp3", ".wav", ".flac", ".ogg"}
FORMATS = "MP3, WAV, FLAC and OGG"

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
            detail=f"That file could not be decoded. {FORMATS} work.",
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


# Async with nothing to await, so it is answered on the event loop and never
# waits behind the worker threads. Render restarts the service when this is
# slow, so it must not depend on how busy the analysis is.
@app.get("/api/health")
async def health():
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

# One analysis at a time. One analysis already plateaus near 440 MB on a
# 512 MB instance, so two overlapping uploads were enough to get the process
# killed for everyone.
_analysis_slot = threading.BoundedSemaphore(1)
ANALYSIS_WAIT_SECONDS = 30

# How many uploads may wait for the slot; any more are refused at once. They
# wait on the event loop, not in a worker thread: when each waiter parked a
# thread, forty of them filled the pool and /api/health went unanswered for
# thirty seconds.
ANALYSIS_QUEUE = 2
# The running analysis plus those waiting for the slot.
_analysis_admitted = 0
_admission_lock = threading.Lock()
# Polled rather than awaited on an asyncio primitive, which binds itself to
# the first event loop that waits on it. One loop serves production, but the
# test client runs each request on a fresh one.
ANALYSIS_POLL_SECONDS = 0.1
# One ticket per waiting upload, in arrival order. Only the head may take the
# slot: when every waiter polled for it, whichever woke first after a release
# won, and an early upload could run out its deadline behind later ones.
_waiting = deque()


def _busy():
    return HTTPException(
        status_code=503,
        detail="Someone else's song is being analysed. Try again in a "
               "few seconds.",
        headers={"Retry-After": "10"},
    )


def _too_large():
    return HTTPException(
        status_code=413,
        detail="That file is over the 25 MB limit, which is a full "
               "length song at a normal bitrate.",
    )


def _queue_full():
    with _admission_lock:
        return _analysis_admitted >= 1 + ANALYSIS_QUEUE


def _admit():
    global _analysis_admitted
    with _admission_lock:
        if _analysis_admitted >= 1 + ANALYSIS_QUEUE:
            return False
        _analysis_admitted += 1
        return True


def _leave():
    global _analysis_admitted
    with _admission_lock:
        _analysis_admitted -= 1


async def _take_slot():
    ticket = object()
    with _admission_lock:
        _waiting.append(ticket)
    deadline = time.monotonic() + ANALYSIS_WAIT_SECONDS
    try:
        while True:
            with _admission_lock:
                if _waiting[0] is ticket and _analysis_slot.acquire(blocking=False):
                    _waiting.popleft()
                    return
            if time.monotonic() >= deadline:
                raise _busy()
            await asyncio.sleep(ANALYSIS_POLL_SECONDS)
    except BaseException:
        # A waiter that gives up, times out or is cancelled must leave the
        # line, or everyone behind it waits for a head that never moves.
        with _admission_lock:
            _waiting.remove(ticket)
        raise


async def _until_done(fn, *args):
    """Run fn in a worker thread and return only once that thread has ended.

    A thread cannot be stopped, and run_in_threadpool returns as soon as its
    task is cancelled, so a cancelled request would otherwise free the slot
    and delete its upload while the analysis was still running on both.
    Cancellation is held back until the thread finishes, then re-raised.
    """
    work = asyncio.ensure_future(run_in_threadpool(fn, *args))
    cancelled = None
    while not work.done():
        try:
            await asyncio.wait({work})
        except asyncio.CancelledError as error:
            cancelled = error
    if cancelled is not None:
        raise cancelled
    return work.result()


def _capped(request: Request):
    """The same request, refused once its body passes MAX_BODY_BYTES.

    Content-Length is checked first but is only a claim, and a chunked upload
    sends none, so the cap is also applied to the bytes as they arrive.
    Without it the multipart parser spools a body of any size to disk.
    """
    received = 0

    async def receive():
        nonlocal received
        message = await request.receive()
        received += len(message.get("body", b""))
        if received > MAX_BODY_BYTES:
            raise _too_large()
        return message

    return Request(request.scope, receive)


def _audio_only(request: Request):
    """The same request, refused as soon as the file's part headers name a
    format this does not read.

    The form parser writes a part's body to a spooled file before the handler
    can see its name, so a check made afterwards let a .txt of up to 25 MB
    arrive and be spooled before the 415. This reads the part headers as they
    stream in, which costs milliseconds even over a full 25 MB body.
    """
    content_type, params = parse_options_header(request.headers.get("content-type", ""))
    boundary = params.get(b"boundary")
    # Anything else gives the form no file at all, which is already a 422.
    if content_type != b"multipart/form-data" or not boundary:
        return request

    header = {"name": b"", "value": b"", "disposition": b""}

    def on_header_field(data, start, end):
        header["name"] += data[start:end]

    def on_header_value(data, start, end):
        header["value"] += data[start:end]

    def on_header_end():
        if header["name"].lower() == b"content-disposition":
            header["disposition"] = header["value"]
        header["name"] = header["value"] = b""

    def on_headers_finished():
        _, options = parse_options_header(header["disposition"])
        header["disposition"] = b""
        # The form hands the handler the last part named "file", which is not
        # known until the body ends, so every one of them has to pass.
        if options.get(b"name") != b"file" or b"filename" not in options:
            return
        filename = options[b"filename"].decode("utf-8", "replace")
        suffix = os.path.splitext(filename)[1].lower()
        if suffix not in ALLOWED:
            raise HTTPException(
                status_code=415,
                detail=f"{suffix or 'A file with no extension'} is not an audio "
                       f"format this reads. {FORMATS} work.",
            )

    parser = MultipartParser(boundary, {
        "on_header_field": on_header_field,
        "on_header_value": on_header_value,
        "on_header_end": on_header_end,
        "on_headers_finished": on_headers_finished,
    })

    async def receive():
        message = await request.receive()
        parser.write(message.get("body", b""))
        return message

    return Request(request.scope, receive)


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


# Declared by hand because the body is parsed by hand, below, so that the
# cheap refusals come before it rather than after.
UPLOAD_SCHEMA = {"requestBody": {"required": True, "content": {"multipart/form-data": {
    "schema": {"type": "object", "required": ["file"],
               "properties": {"file": {"type": "string", "format": "binary"}}}}}}}


@app.post("/api/analyse", openapi_extra=UPLOAD_SCHEMA)
async def analyse(request: Request):
    """Admission on the event loop, analysis in one worker thread.

    Decoding, YAMNet and the spectrogram are blocking CPU work. Run on the
    event loop, nothing else is served for the duration, including
    /api/health. On a host that health checks every few seconds and restarts
    on failure, that turns every analysis into a restart, which is exactly
    what it did. So that work goes to a worker thread, and only the upload
    holding the analysis slot ever occupies one.
    """
    _rate_limit(request)

    # Both before the body is read: an oversized upload by what it declares,
    # and any upload at all when it would only be refused once it had arrived.
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise _too_large()
    if _queue_full():
        raise _busy()

    # Closing the form deletes its spooled temporary file, on every path out.
    # The file's name is checked as its part headers arrive, so a wrong one is
    # refused without the rest of the body being read. An extensionless upload
    # once went straight to the audio decoders.
    async with _audio_only(_capped(request)).form() as form:
        file = form.get("file")
        if not isinstance(file, UploadFile):
            raise HTTPException(status_code=422, detail="No file was attached.")

        # Before admission, so a file the analysis would refuse does not wait
        # for the slot first.
        if file.size is not None and file.size > MAX_BYTES:
            raise _too_large()

        if not _admit():
            raise _busy()
        try:
            await _take_slot()
            try:
                # Waits out the thread even if this request is cancelled, so
                # the slot and the spooled upload are released only once the
                # analysis has stopped using them.
                return await _until_done(_analyse_upload, file)
            finally:
                _analysis_slot.release()
        finally:
            _leave()


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
            raise _too_large()
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
