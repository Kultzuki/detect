# TruthLens (detect)

TruthLens is a FastAPI service for deepfake detection on **images** and **videos**.  
It preprocesses media, runs three model pipelines, and combines outputs with a weighted ensemble to return a final verdict.

## What it does

- Accepts uploaded media via `POST /analyze`
- Supports:
  - Images: `.jpg`, `.jpeg`, `.png`, `.webp`
  - Videos: `.mp4`, `.mov`, `.avi`
- Performs face-focused preprocessing (MTCNN + fallback center crop)
- Runs three detectors:
  - `prithivMLmods/Deep-Fake-Detector-v2-Model` (ViT)
  - `yermandy/deepfake-detection` (CLIP/torchscript path)
  - `prithivMLmods/deepfake-detector-model-v1` (SigLIP)
- Aggregates model scores with ensemble weights:
  - deepfake_v2: `0.30`
  - clip: `0.45`
  - siglip: `0.25`

## Repository layout

```text
truthlens/
  api/
    main.py        # FastAPI app, lifespan warmup, CORS, /health
    router.py      # /analyze endpoint orchestration
    schemas.py     # Response models
  models/
    loader.py      # Model loading/caching + VRAM guard
    ensemble.py    # Weighted final scoring
  pipelines/
    preprocessor.py    # file type detect, image/video preprocessing, frame extraction
    image_pipeline.py  # ViT image inference
    video_pipeline.py  # ViT batched frame inference
    clip_pipeline.py   # CLIP pipeline
    siglip_pipeline.py # SigLIP pipeline
```

## Installation

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r truthlens/requirements.txt
```

> Note: `ffmpeg-python` requires FFmpeg to be available on the system.

## Run the API

From repository root:

```bash
uvicorn truthlens.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

## API usage

### `POST /analyze`

- Content type: `multipart/form-data`
- Field: `file`
- Max upload size: **50MB**

Example:

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@/absolute/path/to/sample.mp4"
```

Response fields include:

- `verdict` (`REAL` or `DEEPFAKE`)
- `confidence_pct` (0–100, derived from distance to 0.5 score)
- `final_score` (ensemble score)
- `per_model` (individual model confidences)
- `processing_time_s`
- `media_type`

## CORS

Set allowed origins with:

```bash
export TRUTHLENS_ALLOWED_ORIGINS="http://localhost:3000,http://127.0.0.1:5173"
```

Default is `*` (all origins).

## Local test UI

Open:

- `truthlens/test_ui.html`

It posts files to `http://localhost:8000/analyze` and displays raw JSON responses.
