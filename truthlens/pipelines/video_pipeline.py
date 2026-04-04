from typing import List, Dict, Any

import numpy as np
import torch
from PIL import Image

from truthlens.models.loader import ModelLoader, load_deepfake_v2
from truthlens.pipelines.utils import resolve_deepfake_index


def run_video_pipeline(frames: List[np.ndarray], loader: ModelLoader) -> Dict[str, Any]:
    """
    Video pipeline: runs Deep-Fake-Detector-v2 on extracted frames.
    Processes up to 40 provided frames via batched inference.
    """
    model, processor = loader.load("deepfake_v2", load_deepfake_v2)
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    id2label = getattr(model.config, "id2label", {})
    deepfake_idx = resolve_deepfake_index(id2label)

    # Frames are already downsampled upstream; avoid an extra skip that drops signal.
    pil_frames = [Image.fromarray(frame) for frame in frames[:40]]
                
    if not pil_frames:
        return {
            "deepfake_confidence": 0.0,
            "model": "prithivMLmods/Deep-Fake-Detector-v2-Model"
        }

    # Batch process all frames simultaneously
    inputs = processor(images=pil_frames, return_tensors="pt")
    
    # Cast to correct device/dataType (fp16 support)
    inputs = {
        k: v.to(device, dtype=model_dtype) if torch.is_floating_point(v) else v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        out = model(**inputs)
        probs = torch.softmax(out.logits, dim=1)

    # Extract score for all frames rapidly
    raw_scores = probs[:, deepfake_idx].cpu().numpy()
    
    # Aggregation rule: 75th percentile removes low-confidence/garbage frames
    # while boosting the "most suspicious" moments (much stronger than mean matching)
    final_score = np.percentile(raw_scores, 75)
    
    # Clip for absolute confidence realism
    final_score = np.clip(final_score, 0.05, 0.95)

    return {
        "deepfake_confidence": float(final_score),
        "frame_scores": raw_scores.tolist(),
        "model": "prithivMLmods/Deep-Fake-Detector-v2-Model"
    }
