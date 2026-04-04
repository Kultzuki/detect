import numpy as np
import torch
from PIL import Image

from truthlens.models.loader import ModelLoader, load_clip_deepfake


def run_clip_pipeline(frame_or_frames, loader: ModelLoader) -> dict:
    model, processor = loader.load("clip_deepfake", load_clip_deepfake)

    # Accept single frame (image) or list (video)
    if isinstance(frame_or_frames, list):
        pil_images = [Image.fromarray(f) for f in frame_or_frames[::3][:40]]
    else:
        pil_images = [Image.fromarray(frame_or_frames)]

    if not pil_images:
        return {
            "deepfake_confidence": 0.5,
            "model": "yermandy/deepfake-detection"
        }

    inputs = processor(images=pil_images, return_tensors="pt")

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    inputs = {
        k: v.to(device, dtype=model_dtype) if torch.is_floating_point(v) else v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        # Torchscript format for yermandy/deepfake-detection.
        if isinstance(model, torch.jit.ScriptModule):
            logits = model(inputs["pixel_values"])
        else:
            output = model(**inputs)
            if hasattr(output, "logits"):
                logits = output.logits
            elif torch.is_tensor(output):
                logits = output
            else:
                # Last fallback if output is not a standard classification container.
                image_features = model.get_image_features(**inputs)
                if image_features.ndim != 2 or image_features.shape[1] < 2:
                    raise RuntimeError("CLIP output does not expose 2-class logits")
                logits = image_features[:, :2]

        probs = torch.softmax(logits, dim=-1)

    # yermandy torchscript inference defines outputs as (p_real, p_fake)
    fake_idx = 1
    scores = probs[:, fake_idx].detach().float().cpu().numpy()
    final = float(np.percentile(scores, 75)) if len(scores) > 1 else float(scores[0])
    final = float(np.clip(final, 0.05, 0.95))

    return {
        "deepfake_confidence": final,
        "model": "yermandy/deepfake-detection"
    }
