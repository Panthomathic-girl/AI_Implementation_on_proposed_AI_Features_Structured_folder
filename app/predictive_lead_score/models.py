# app/predictive_lead_score/models.py
from typing import TypedDict, List
from pydantic import BaseModel, Field

class LeadScoreResponse(BaseModel):
    score: int = Field(..., ge=0, le=100)
    explanation: str
    # transcription: str
    filename: str


class LeadAnalysisResult(TypedDict):
    transcription: str
    score: int
    explanation: str


class LeadInput(BaseModel):
    lead_id: str
    vertical: str
    territory: str
    lead_source: str
    product_stage: str
    target_price: float
    proposed_price: float
    price_discount_pct: float
    expected_order_volume: float
    expected_frequency: str
    hod_approval: int
    emails_sent: int
    emails_opened: int
    calls_made: int
    meetings_held: int
    avg_response_time_hours: float
    last_contact_age_days: int
    complaint_logged: int
    buying_trend_percent: float
    previous_orders: int
    inactive_flag: int
    overdue_payments: int
    license_expiry_days_left: int
    training_completed: int
    deal_age_days: int

# === Batch Request Model ===
class BatchPredictionRequest(BaseModel):
    leads: List[LeadInput]

# === Batch Response Item ===
class BatchPredictionResponse(BaseModel):
    lead_id: str
    prediction: str
    score: float
