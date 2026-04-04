import os
import time
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException

from truthlens.pipelines.preprocessor import preprocess
from truthlens.pipelines.image_pipeline import run_image_pipeline
from truthlens.pipelines.video_pipeline import run_video_pipeline
from truthlens.pipelines.clip_pipeline import run_clip_pipeline
from truthlens.pipelines.siglip_pipeline import run_siglip_pipeline
from truthlens.models.ensemble import EnsembleEngine
from truthlens.models.loader import ModelLoader
from truthlens.api.schemas import AnalysisResult, ModelScore

router = APIRouter()

# Instantiate Singletons
loader = ModelLoader()
engine = EnsembleEngine()

@router.post("/analyze", response_model=AnalysisResult)
async def analyze_media(file: UploadFile = File(...)):
    start_time = time.time()
    
    # 1. Validate max 50MB size without loading fully into memory
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")
        
    temp_dir = f"/tmp/truthlens_uploads/{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)
    safe_name = file.filename or "upload.bin"
    file_path = os.path.join(temp_dir, safe_name)
    
    try:
        # 2. Save UploadFile
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 3. Route through preprocessor logic
        try:
            prep_result = preprocess(file_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")
            
        media_type = prep_result["type"]
        frames = prep_result.get("frames")
        if frames is None:
            raise HTTPException(status_code=400, detail="Preprocessing failed: no visual frames extracted")
        
        # 4. Pipeline orchestration mapping based on media_type
        pipeline_scores = {}
        
        if media_type == "image":
            pipeline_scores["deepfake_v2"] = run_image_pipeline(frames, loader)
            pipeline_scores["clip"] = run_clip_pipeline(frames, loader)
            pipeline_scores["siglip"] = run_siglip_pipeline(frames, loader)
            
        elif media_type == "video":
            pipeline_scores["deepfake_v2"] = run_video_pipeline(frames, loader)
            pipeline_scores["clip"] = run_clip_pipeline(frames, loader)
            pipeline_scores["siglip"] = run_siglip_pipeline(frames, loader)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported media type: {media_type}")
            
        # 5. Determine Verdict via EnsembleEngine
        ensemble_result = engine.compute(pipeline_scores)
        
        # Map output keys to Pydantic objects for strong type formatting
        per_model_list = [
            ModelScore(model_name=m, confidence=c)
            for m, c in ensemble_result["per_model"].items()
        ]
        
        processing_time = float(time.time() - start_time)
        
        # 6. Return JSON Context
        return AnalysisResult(
            verdict=ensemble_result["verdict"],
            confidence_pct=ensemble_result["confidence_pct"],
            final_score=ensemble_result["final_score"],
            per_model=per_model_list,
            heatmap_url=None,    # Opted-out
            report_url=None,     # Not implemented per opt-out structure
            processing_time_s=processing_time,
            media_type=media_type
        )
        
    finally:
        # Secure cleanup ensuring HDD/SSD isn't spammed with residual payload artifacts
        shutil.rmtree(temp_dir, ignore_errors=True)
