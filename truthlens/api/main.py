import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch

from truthlens.api.router import router, loader
from truthlens.models.loader import load_deepfake_v2, load_clip_deepfake, load_siglip_deepfake


def _parse_allowed_origins() -> list[str]:
    raw = os.getenv("TRUTHLENS_ALLOWED_ORIGINS", "*").strip()
    if raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


ALLOWED_ORIGINS = _parse_allowed_origins()
ALLOW_CREDENTIALS = "*" not in ALLOWED_ORIGINS


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("TruthLens initialized. Pre-warming deepfake_v2, clip_deepfake, and siglip_deepfake...")
    loader.load("deepfake_v2", load_deepfake_v2)
    loader.load("clip_deepfake", load_clip_deepfake)
    loader.load("siglip_deepfake", load_siglip_deepfake)
    print("All 3 models warm.")
    yield

# 1. Initialize FastAPI Application
app = FastAPI(title="TruthLens API", version="1.0", lifespan=lifespan)

# 2. Assign CORS Middleware for dev isolation/frontend interfacing
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Include Core Domain Routing
app.include_router(router)

# 4. Standardized Health Probe Endpoint
@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "gpu": torch.cuda.is_available()
    }
