# app/smart_task_prioritization/models.py
from pydantic import BaseModel, Field
from typing import List
from config import TASKTYPE


class SuggestedTask(BaseModel):
    id: int = Field(..., ge=1, description="Sequential task ID: 1, 2, 3...")
    task_type: TASKTYPE
    explanation: str = Field(..., min_length=10, description="Clear, actionable next step with context")

class SmartTaskResponse(BaseModel):
    tasks: List[SuggestedTask] = Field(default=[], description="List of suggested post-call tasks")
    call_has_next_steps: bool = Field(default=True, description="True if tasks were found")