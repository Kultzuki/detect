from typing import Any, Mapping


def resolve_deepfake_index(id2label: Mapping[Any, Any]) -> int:
    """Resolve the deepfake class index from model config labels with safe fallbacks."""
    normalized = {}
    for raw_idx, label in id2label.items():
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        normalized[idx] = str(label).strip().lower()

    for idx, label in normalized.items():
        if label in {"deepfake", "fake", "ai-generated", "ai generated", "synthetic"}:
            return idx

    if len(normalized) == 2:
        for idx, label in normalized.items():
            if label in {"real", "realism", "authentic", "genuine"}:
                return 1 - idx

    # Conservative fallback for unknown binary label spaces.
    return 1
