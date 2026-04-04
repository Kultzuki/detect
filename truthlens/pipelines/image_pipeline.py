import numpy as np
from PIL import Image
import torch

from truthlens.models.loader import ModelLoader, load_deepfake_v2
from truthlens.pipelines.utils import resolve_deepfake_index


def run_image_pipeline(frame: np.ndarray, loader: ModelLoader) -> dict:
    pil_image = Image.fromarray(frame)

    model, processor = loader.load("deepfake_v2", load_deepfake_v2)

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    
    # Process image (even for single image, pass as list for consistency)
    inputs = processor(images=[pil_image], return_tensors="pt")
    
    # Cast to model device and float precision (fp16)
    inputs = {
        k: v.to(device, dtype=model_dtype) if torch.is_floating_point(v) else v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    id2label = getattr(model.config, "id2label", {})
    deepfake_idx = resolve_deepfake_index(id2label)
    
    # Extract prediction from batch dimension 0
    deepfake_score = probs[0][deepfake_idx].item()
    
    # Clamp score logically (avoid 1.0 or 0.0 false total certainty)
    deepfake_score = np.clip(deepfake_score, 0.05, 0.95)

    return {
        "deepfake_confidence": float(deepfake_score),
        "model": "prithivMLmods/Deep-Fake-Detector-v2-Model"
    }
