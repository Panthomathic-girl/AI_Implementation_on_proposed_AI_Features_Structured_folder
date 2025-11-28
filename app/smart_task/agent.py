# app/smart_task_prioritization/agent.py
from app.llm import analyze_audio_file
from app.smart_task.utils import parse_task_suggestion


def suggest_next_tasks(file_path: str, mime_type: str) -> dict:
    prompt = TASK_SUGGESTION_PROMPT = """
You are the best B2B sales assistant in the world. Your only job: listen to this call and tell the rep exactly what to do next.

Rules:
- Find 1–3 concrete next actions
- Even if the prospect is vague, suggest the most logical follow-up
- Never say "no action needed" unless the deal is clearly dead
- Be specific: dates, discounts, stakeholders, documents

Return ONLY valid JSON. No explanations. No markdown.

Correct format:
{
  "tasks": [
    {"id": 1, "task_type": "scheduling", "explanation": "Schedule demo call next Tuesday at 2 PM – prospect wants technical team involved."},
    {"id": 2, "task_type": "send_proposal", "explanation": "Send proposal with 12-month contract and 8% early payment discount."},
    {"id": 3, "task_type": "follow_up_email", "explanation": "Send meeting recap and attach case study from similar hospital."}
  ]
}

Only use these task_type values:
scheduling, send_proposal, follow_up_email, demo_scheduling, internal_review, contract_preparation, payment_followup, feedback_request

If deal is lost → return {"tasks": []}
Otherwise → always return at least 1 task.

Now analyze the call and respond with JSON only.
"""

    # No await — analyze_audio_file returns GenerateContentResponse immediately
    raw_response = analyze_audio_file(file_path, mime_type, prompt)

    # Parse safely with our bulletproof parser
    parsed_result = parse_task_suggestion(raw_response)

    # Return dict for Pydantic model
    return parsed_result.dict()