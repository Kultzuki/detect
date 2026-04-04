from typing import Dict, Any

MODEL_WEIGHTS = {
    "deepfake_v2": 0.30,
    "clip": 0.45,
    "siglip": 0.25,
}


class EnsembleEngine:
    def compute(self, pipeline_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        total_weight = 0.0
        weighted_sum = 0.0
        per_model = {}

        if not pipeline_scores:
            raise ValueError("No valid pipeline scores provided to EnsembleEngine.")

        for key, result in pipeline_scores.items():
            score = float(result.get("deepfake_confidence", 0.5))
            weight = MODEL_WEIGHTS.get(key, 1.0 / len(pipeline_scores))

            weighted_sum += score * weight
            total_weight += weight
            per_model[result.get("model", key)] = score

        final_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        confidence_pct = int(round(abs(final_score - 0.5) * 200))
        verdict = "DEEPFAKE" if final_score >= 0.5 else "REAL"

        return {
            "verdict": verdict,
            "confidence_pct": confidence_pct,
            "final_score": final_score,
            "per_model": per_model,
        }
