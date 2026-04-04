import numpy as np
import torch
from PIL import Image

from truthlens.models.loader import ModelLoader, load_siglip_deepfake

_SIGLIP_LABELS = {"fake": 0, "real": 1}


def run_siglip_pipeline(frame_or_frames, loader: ModelLoader) -> dict:
    model, processor = loader.load("siglip_deepfake", load_siglip_deepfake)
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    if isinstance(frame_or_frames, list):
        raw = [Image.fromarray(f) for f in frame_or_frames[::3][:40]]
    else:
        raw = [Image.fromarray(frame_or_frames)]

    if not raw:
        return {
            "deepfake_confidence": 0.5,
            "model": "prithivMLmods/deepfake-detector-model-v1"
        }

    # Let the processor apply its own configured resize policy.
    inputs = processor(images=raw, return_tensors="pt")
    inputs = {
        k: v.to(device, dtype=model_dtype) if torch.is_floating_point(v) else v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    fake_idx = _SIGLIP_LABELS["fake"]
    scores = probs[:, fake_idx].detach().float().cpu().numpy()
    final = float(np.percentile(scores, 75)) if len(scores) > 1 else float(scores[0])
    final = float(np.clip(final, 0.05, 0.95))

    return {
        "deepfake_confidence": final,
        "model": "prithivMLmods/deepfake-detector-model-v1"
    }
