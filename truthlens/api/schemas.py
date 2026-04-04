from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class ModelScore(BaseModel):
    model_name: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

class AnalysisResult(BaseModel):
    verdict: str
    confidence_pct: int
    final_score: float
    per_model: List[ModelScore]
    heatmap_url: Optional[str] = None
    report_url: Optional[str] = None
    processing_time_s: float
    media_type: str
