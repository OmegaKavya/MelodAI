# Two stages so the Node toolchain does not ship in the running image.
FROM node:20-slim AS web
WORKDIR /build
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
# librosa falls back to audioread for formats libsndfile cannot decode.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs the container as a non-root user, and numba needs a
# writable cache directory or it fails on first import.
RUN useradd -m -u 1000 app
ENV HOME=/home/app \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/mpl \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY server/requirements.txt ./server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

COPY melodai/ ./melodai/
COPY server/ ./server/
COPY models/energy_head.npz models/corpus.npz models/yamnet.tflite models/yamnet_class_map.csv ./models/
COPY --from=web /build/dist ./web/dist

USER app
EXPOSE 7860
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
