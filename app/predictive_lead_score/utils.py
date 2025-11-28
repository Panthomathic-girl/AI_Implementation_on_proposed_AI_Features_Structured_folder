# app/predictive_lead_score/utils.py
from typing import Dict, Any, Optional, TypedDict
from app.predictive_lead_score.models import LeadAnalysisResult
import json
import pandas as pd
import joblib
import os
import shutil
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import ModelConfig
from typing import Any, Optional
import logging

# Optional: If you're using actual protobuf objects from google-generativeai
try:
    from google.generativeai.types import GenerateContentResponse
    PROTO_AVAILABLE = True
except ImportError:
    PROTO_AVAILABLE = False

TARGET_MODEL_PATH = ModelConfig.LEAD_MODEL_FILE


class Finetune:
    @staticmethod
    async def auto_preprocess(df: pd.DataFrame):
        df = df.copy()
        target = next((col for col in ["class", "deal_closed", "won", "target"] if col in df.columns), None)
        if not target:
            raise ValueError("Target column not found (class, deal_closed, won, target)")

        y = df[target].astype(int)
        X = df.drop(columns=[target])

        # Drop ID columns
        X = X.drop(columns=[c for c in X.columns if "id" in c.lower()], errors="ignore")

        # Encode categoricals (same as training)
        for col in X.select_dtypes(include=['object', 'string']).columns:
            X[col] = X[col].fillna("missing")
            X[col] = pd.factorize(X[col])[0]

        X = X.fillna(X.median(numeric_only=True)).fillna(0)
        return X, y

    # ────────────────────── PREPROCESS FOR INFERENCE ──────────────────────
    @staticmethod
    def preprocess_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply EXACTLY the same preprocessing used during training.
        This must be used in the /predict endpoint before calling model.predict_proba().
        """
        df = df.copy()

        # 1. Drop any ID columns (like lead_id)
        df = df.drop(columns=[c for c in df.columns if "id" in c.lower()], errors="ignore")

        # 2. Encode categorical columns exactly like training
        for col in df.select_dtypes(include=['object', 'string']).columns:
            df[col] = df[col].fillna("missing")
            # Use factorize (same as training) — creates integer labels
            df[col] = pd.factorize(df[col])[0]

        # 3. Fill missing numeric values (same as training)
        df = df.fillna(df.median(numeric_only=True)).fillna(0)

        return df

    # ────────────────────── TRAIN & REPLACE ROOT MODEL ──────────────────────
    @classmethod
    async def train_model_from_csv(cls, csv_path: str) -> dict:
        df = pd.read_csv(csv_path)
        X, y = await cls.auto_preprocess(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )

        model = RandomForestClassifier(
            n_estimators=600,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, average='weighted', zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, average='weighted', zero_division=0), 4),
            "f1_score": round(f1_score(y_test, preds, average='weighted'), 4),
            "total_samples": len(df),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Overwrite the production model
        joblib.dump(model, TARGET_MODEL_PATH)

        return {
            "status": "Model retrained & deployed",
            "model_path": TARGET_MODEL_PATH,
            "metrics": metrics,
        }


# ────────────────────── GEMINI RESPONSE PARSING (unchanged) ──────────────────────
def parse_gemini_generate_content_response(response: Any) -> Optional[LeadAnalysisResult]:
    text_content = ""

    if PROTO_AVAILABLE and isinstance(response, GenerateContentResponse):
        if response.candidates and response.candidates[0].content.parts:
            text_content = response.candidates[0].content.parts[0].text

    elif isinstance(response, dict):
        candidates = response.get("candidates", [])
        if candidates and "content" in candidates[0]:
            parts = candidates[0]["content"].get("parts", [])
            if parts and "text" in parts[0]:
                text_content = parts[0]["text"]

    elif isinstance(response, str):
        json_match = re.search(r"```json\s*({.*?})\s*```", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                return {
                    "transcription": parsed.get("transcription", ""),
                    "score": parsed.get("score", 0),
                    "explanation": parsed.get("explanation", "")
                }
            except json.JSONDecodeError:
                pass

        text_match = re.search(r'"text":\s*"([^"]+)"', response.replace('\n', ' '))
        if text_match:
            text_content = text_match.group(1).replace("\\n", "\n")

    if text_content.strip():
        json_block_match = re.search(r"```json\s*({.*?})\s*```", text_content, re.DOTALL)
        if json_block_match:
            try:
                data = json.loads(json_block_match.group(1))
                return {
                    "transcription": data.get("transcription", ""),
                    "score": float(data.get("score", 0)),
                    "explanation": data.get("explanation", "")
                }
            except (json.JSONDecodeError, ValueError):
                return None

        try:
            data = json.loads(text_content.strip())
            return {
                "transcription": data.get("transcription", ""),
                "score": float(data.get("score", 0)),
                "explanation": data.get("explanation", "")
            }
        except json.JSONDecodeError:
            pass

    return None