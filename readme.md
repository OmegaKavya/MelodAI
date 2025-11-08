# MelodAI

Classify music mood from short audio clips by converting them into mel-spectrogram images and training a CNN.

## What this project does

- Downloads 90-second clips for a curated set of songs per mood
- Generates augmented, longer-context enhanced spectrograms (2×45s segments with pitch/time variants)
- Resizes enhanced spectrograms to 224×224 for efficient CNN training
- Trains a 3-class CNN (energetic, peaceful, emotional) and evaluates performance

Folder snapshot (key dirs):

- `data/` — raw 90s audio clips organized by mood (not committed)
- `mel_spectrograms_enhanced/` — augmented 45s segment spectrograms (original + pitch/time variants)
- `mel_spectrograms_resized/` — 224×224 versions used for training
- `models/` — saved Keras models and training artifacts
- `scripts/` — CLI scripts for data prep, training, and evaluation

## Prerequisites

- Python 3.9–3.11
- macOS or Windows 10/11
- System dependencies:
  - ffmpeg (required by pydub)
  - yt-dlp CLI (for downloading audio)

### macOS setup

Install system tools:

```bash
brew install ffmpeg yt-dlp
```

Apple Silicon (M1/M2) TensorFlow options:

- Preferred: use `tensorflow-macos` and `tensorflow-metal` for GPU acceleration.
- Alternative: `tensorflow` CPU-only works but will be slower.

### Windows setup

1. Install Python from https://www.python.org/downloads/ (enable "Add python.exe to PATH").
2. Install ffmpeg:

- Download static build from https://www.gyan.dev/ffmpeg/builds/
- Extract and add the `bin/` folder path to System Environment Variable `Path`.

3. Install yt-dlp:

```powershell
pip install yt-dlp
```

(Or download a Windows binary from https://github.com/yt-dlp/yt-dlp/releases and place it in PATH.) 4. (Optional GPU) Install appropriate TensorFlow build for your GPU drivers (CUDA 12.x). Otherwise use CPU. 5. Verify tools:

```powershell
ffmpeg -version
yt-dlp --version
python --version
```

## Setup

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Apple Silicon (GPU acceleration):

```bash
pip uninstall -y tensorflow
pip install tensorflow-macos tensorflow-metal
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Deactivate environment:

```bash
deactivate
```

## Typical workflow

1. Fetch YouTube URLs (optional)

- Script: `scripts/fetch_url.py`
- Requires a YouTube Data API v3 key from Google Cloud.
- Edit `API_KEY` in the script or export an env var and read it inside.

```bash
python scripts/fetch_url.py
```

2. Download audio (90s per song)

- Script: `scripts/download_songs.py`
- Requires `yt-dlp` and `ffmpeg` to be installed and on PATH.
- Outputs 90s MP3s into `data/<mood>/`.

macOS / Linux:

```bash
python scripts/download_songs.py
```

Windows PowerShell:

```powershell
python scripts/download_songs.py
```

3. Generate enhanced spectrograms (with augmentation)

- Script: `scripts/generate_smart_segments.py`
- Creates 2×45s segments per song and applies augmentations (pitch shift, time stretch)
- Outputs to `mel_spectrograms_enhanced/`

```bash
python scripts/generate_smart_segments.py
```

4. Resize spectrograms for training

- Script: `scripts/resizing.py`
- Inputs from `mel_spectrograms_enhanced/` and outputs `mel_spectrograms_resized/` (224×224).

```bash
python scripts/resizing.py
```

5. Train the model (3 classes)

- Script: `scripts/train_segmented_model.py`
- Uses `mel_spectrograms_resized/` (falls back to `mel_spectrograms_enhanced/` if not resized yet)
- Saves best and final models in `models/`.

```bash
python scripts/train_segmented_model.py
```

6. Evaluate the model

- Script: `scripts/evaluate_model.py`
- Generates a confusion matrix and detailed metrics in `models/`.

```bash
python scripts/evaluate_model.py
```

## Data layout expectations

- Enhanced (augmented) spectrograms:

  - `mel_spectrograms_enhanced/energetic/*_seg{1,2}_{original|pitch_up|pitch_down|time_stretch}.png`
  - `mel_spectrograms_enhanced/peaceful/...`
  - `mel_spectrograms_enhanced/emotional/...`

- Resized training set:
  - `mel_spectrograms_resized/energetic/*.png`
  - `mel_spectrograms_resized/peaceful/*.png`
  - `mel_spectrograms_resized/emotional/*.png`

## Tips and troubleshooting

- ffmpeg not found:
  - macOS: `brew install ffmpeg`
  - Windows: ensure `ffmpeg\bin` path is in System PATH (restart terminal)
- yt-dlp not found: `pip install yt-dlp` (or brew on macOS)
- TensorFlow GPU (Apple Silicon): install `tensorflow-macos tensorflow-metal`
- TensorFlow GPU (Windows NVIDIA): install CUDA toolkit + cuDNN (match TensorFlow version) or stick with CPU if unsure
- Too few spectrograms? Rerun `generate_smart_segments.py` after adding more audio.
- Memory errors during training: lower batch size in `train_segmented_model.py` (e.g., 8 or 4)
- OpenCV import errors: `pip install --upgrade opencv-python`
- Google API quota errors: slow requests, or rotate API key
- Mismatched Python env: ensure venv is activated before running scripts
