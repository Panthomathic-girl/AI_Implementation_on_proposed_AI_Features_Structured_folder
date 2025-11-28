# app/smart_task_prioritization/config.py
import json
import re
import logging
from typing import Optional, Dict, Any, Literal
from .models import SmartTaskResponse

logger = logging.getLogger(__name__)



def extract_json_from_llm_response(text: str) -> Optional[Dict]:
    """Extract JSON from Gemini response – handles code blocks, malformed output, etc."""
    if not text:
        return None

    # Try ```json ... ``` block
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            logger.warning(f"JSON block found but invalid: {e}")

    # Try raw JSON
    try:
        cleaned = text.strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            return json.loads(cleaned)
    except:
        pass

    # Last resort: look for { "tasks": ... }
    rough_match = re.search(r'\{.*"tasks"\s*:\s*\[.*\].*}', text, re.DOTALL)
    if rough_match:
        try:
            return json.loads(rough_match.group(0))
        except:
            pass

    logger.warning("Could not extract valid JSON from LLM response")
    return None

def parse_task_suggestion(response: Any) -> SmartTaskResponse:
    """Parse Gemini response → SmartTaskResponse (bulletproof like your train_model.py)"""
    try:
        text = ""

        # Handle Gemini protobuf
        if hasattr(response, "text"):
            text = response.text
        elif hasattr(response, "candidates") and response.candidates:
            part = response.candidates[0].content.parts[0]
            text = getattr(part, "text", "")
        elif isinstance(response, dict):
            candidates = response.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    text = parts[0].get("text", "")
        elif isinstance(response, str):
            text = response

        if not text or "tasks" not in text:
            return SmartTaskResponse(tasks=[], call_has_next_steps=False)

        data = extract_json_from_llm_response(text)
        if not data or "tasks" not in data:
            return SmartTaskResponse(tasks=[], call_has_next_steps=False)

        tasks = data["tasks"]
        if not isinstance(tasks, list):
            return SmartTaskResponse(tasks=[], call_has_next_steps=False)

        validated_tasks = []
        for i, task in enumerate(tasks[:5], start=1):  # Max 5 tasks
            if isinstance(task, dict):
                task_id = task.get("id", i)
                task_type = task.get("task_type", "scheduling")
                explanation = task.get("explanation", "").strip()
                if explanation and len(explanation) >= 10:
                    validated_tasks.append({
                        "id": task_id,
                        "task_type": task_type,
                        "explanation": explanation
                    })

        return SmartTaskResponse(
            tasks=validated_tasks,
            call_has_next_steps=len(validated_tasks) > 0
        )

    except Exception as e:
        logger.error(f"Failed to parse task suggestion: {e}")
        return SmartTaskResponse(tasks=[], call_has_next_steps=False)