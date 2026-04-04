import gc
import torch
from typing import Callable, Any, Tuple
from huggingface_hub import hf_hub_download
from transformers import (
    AutoImageProcessor,
    CLIPModel,
    CLIPProcessor,
    SiglipForImageClassification,
    ViTForImageClassification,
    ViTImageProcessor
)

class ModelLoader:
    """
    Multi-model cache for keeping frequently used model/processor pairs loaded.
    This avoids repeated unload/reload thrashing when multiple pipelines run per request.
    """
    def __init__(self):
        self._cache: dict[str, Any] = {}

    @property
    def vram_used_mb(self) -> float:
        """Returns current GPU memory usage in MB via torch.cuda.memory_allocated()"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0

    def load(self, model_name: str, load_fn: Callable) -> Any:
        """
        Loads and caches a model tuple (e.g. model, processor) by name.
        Returns cached objects on subsequent calls.
        """
        if model_name in self._cache:
            return self._cache[model_name]

        loaded = load_fn()
        self._cache[model_name] = loaded

        # Safety constraint (RTX 3050 4GB limit)
        if self.vram_used_mb > 3500:
            self.unload(model_name)
            raise RuntimeError(
                f"VRAM usage exceeded 3500MB after loading {model_name}: {self.vram_used_mb:.2f}MB"
            )

        return loaded

    def unload(self, model_name: str | None = None) -> None:
        """Unload one cached model or clear all cached models."""
        if model_name is None:
            self._cache.clear()
        else:
            self._cache.pop(model_name, None)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ==========================================
# Load Functions for Validated AI Models
# All moved to CUDA, evaluated, and fps16
# ==========================================

def load_deepfake_v2() -> Tuple[Any, Any]:
    """Loads prithivMLmods/Deep-Fake-Detector-v2-Model (ViT-based, primary image model)."""
    model_id = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processor = ViTImageProcessor.from_pretrained(model_id)
    model = ViTForImageClassification.from_pretrained(model_id)
    
    # Save 50% VRAM on GPU setups (+ minor speedup)
    if device.type == "cuda":
        model = model.half()
        
    model = torch.nn.Module.to(model, device)
    model.eval()
    
    return model, processor


def load_clip_deepfake(model_path: str = "yermandy/deepfake-detection") -> Tuple[Any, Any]:
    """
    Loads CLIP-based deepfake detector.

    Note:
    - Some checkpoints (e.g. yermandy/deepfake-detection) are distributed as torchscript
      and do not expose a standard Transformers CLIPModel config/processor pair.
    - This loader first tries CLIPModel + CLIPProcessor as requested, then falls back
      to the torchscript release format for compatibility.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path == "yermandy/deepfake-detection":
        model_path_local = hf_hub_download(repo_id=model_path, filename="model.torchscript")
        model = torch.jit.load(model_path_local, map_location=str(device))
        model.eval()

        if device.type == "cuda":
            model = model.to(torch.bfloat16)

        model = torch.nn.Module.to(model, device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        return model, processor

    try:
        processor = CLIPProcessor.from_pretrained(model_path)
        model = CLIPModel.from_pretrained(model_path)

        if device.type == "cuda":
            model = model.half()

        model = torch.nn.Module.to(model, device)
        model.eval()
        return model, processor
    except Exception:
        # Compatibility path for yermandy/deepfake-detection torchscript artifact.
        model_path_local = hf_hub_download(repo_id=model_path, filename="model.torchscript")
        model = torch.jit.load(model_path_local, map_location=str(device))
        model.eval()

        if device.type == "cuda":
            model = model.to(torch.bfloat16)

        model = torch.nn.Module.to(model, device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        return model, processor


def load_siglip_deepfake(model_path: str = "prithivMLmods/deepfake-detector-model-v1") -> Tuple[Any, Any]:
    """Loads SigLIP deepfake classifier checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(model_path)
    model = SiglipForImageClassification.from_pretrained(model_path)

    if device.type == "cuda":
        model = model.half()

    model = torch.nn.Module.to(model, device)
    model.eval()
    return model, processor
