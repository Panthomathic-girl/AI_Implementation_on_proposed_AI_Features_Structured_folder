# app/predictive_lead_score/views.py
import shutil, os
import uuid
from typing import List
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from .models import LeadScoreResponse, BatchPredictionRequest, BatchPredictionResponse
from app.predictive_lead_score.agent import lead_score_agent
from app.predictive_lead_score.utils import Finetune
import joblib
from config import ModelConfig
router = APIRouter(prefix="/lead", tags=["Lead Scoring"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# FIXED: Added "audio/wave"
ALLOWED_MIME = {
    "audio/mp3", "audio/mpeg",
    "audio/wav", "audio/x-wav", "audio/wave",   # <-- ADDED
    "audio/flac", "audio/x-flac",
    "audio/aac",
    "audio/ogg",
    "audio/webm",
}
ALLOWED_EXT = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".webm"}

MAX_SIZE = 20 * 1024 * 1024  # 20 MB
pipeline = joblib.load(ModelConfig.LEAD_MODEL_FILE)

def predict_single(prob: float) -> str:
    return "Likely to Close" if prob >= 0.5 else "Unlikely to Close"

@router.post("/predict", response_model=List[BatchPredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    try:
        if not request.leads:
            raise HTTPException(status_code=400, detail="No leads provided")

        # Convert to DataFrame
        df_raw = pd.DataFrame([lead.dict() for lead in request.leads])

        # Keep lead_id for response
        lead_ids = df_raw["lead_id"].tolist()
        df_for_model = df_raw.drop(columns=["lead_id"], errors="ignore")

        # APPLY SAME PREPROCESSING AS TRAINING
        df_processed = Finetune.preprocess_for_prediction(df_for_model)

        # Ensure column order matches training (critical!)
        if hasattr(pipeline, "feature_names_in_"):
            expected_cols = list(pipeline.feature_names_in_)
            df_processed = df_processed.reindex(columns=expected_cols, fill_value=0)
        else:
            # Fallback: just use whatever columns we have
            pass

        # Predict
        probabilities = pipeline.predict_proba(df_processed)[:, 1]

        results = [
            BatchPredictionResponse(
                lead_id=lead_id,
                prediction="Likely to Close" if prob >= 0.5 else "Unlikely to Close",
                score=round(prob * 100, 2)  # Return as 0-100 like audio endpoint
            )
            for lead_id, prob in zip(lead_ids, probabilities)
        ]

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/score", response_model=LeadScoreResponse)
async def predict_lead_score(audio: UploadFile = File(...)):
    filename = audio.filename or "unknown"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    if audio.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=400, detail=f"Invalid MIME: {audio.content_type}")

    content = await audio.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="File too large (>20MB)")

    file_id = str(uuid.uuid4())
    temp_path = UPLOAD_DIR / f"{file_id}{ext}"

    try:
        temp_path.write_bytes(content)
        result = await lead_score_agent(str(temp_path), audio.content_type)
        return LeadScoreResponse(**result, filename=filename)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass


@router.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only .csv files allowed")
    
    temp = f"temp_{file.filename}"
    with open(temp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        result = await Finetune.train_model_from_csv(temp)
        return result
    finally:
        if os.path.exists(temp):
            os.remove(temp)