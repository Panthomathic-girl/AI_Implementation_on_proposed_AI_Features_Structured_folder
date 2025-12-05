# config.py
from pathlib import Path
from dotenv import load_dotenv
import os
from typing import Literal


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

class Settings:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    MAX_AUDIO_SIZE_MB: int = int(os.getenv("MAX_AUDIO_SIZE_MB", "20"))
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    

    def __post_init__(self):
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY required in .env")

class ModelConfig:
    LEAD_MODEL_FILE = "app/predictive_lead_score/models/deal_closure_model.pkl"
    ORDER_FORECAST_MODEL_FILE = Path("app/order_forecasting/models/order_forecast_model.pkl") 
    

TASKTYPE = Literal["call", "email", "meeting", "administrative", "follow up"]
PriorityType = Literal["low", "medium", "high", "urgent"]
StatusType = Literal["not started", "in progress", "completed", "waiting", "deferred"]

TASK_TYPE_MAPPING = {
    "internal meeting": "meeting",
    "internal task": "administrative",
    "demo": "meeting",
    "presentation": "meeting",
    "follow-up": "follow up",
    "followup": "follow up",
    "send email": "email",
    "make a call": "call",
    "schedule call": "call",
    "admin": "administrative",
    "administrative task": "administrative",
}

settings = Settings()
