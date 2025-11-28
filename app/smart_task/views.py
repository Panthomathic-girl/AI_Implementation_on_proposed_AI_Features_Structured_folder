# app/smart_task_prioritization/views.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import uuid
import shutil
from app.smart_task.agent import suggest_next_tasks
from app.smart_task.models import SmartTaskResponse

router = APIRouter(prefix="/smart-task", tags=["Smart Task Prioritization"])

UPLOAD_DIR = Path("uploads/smart_tasks")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".aac", ".m4a"}
ALLOWED_MIME = {
    "audio/mp3", "audio/mpeg", "audio/wav", "audio/x-wav", "audio/wave",
    "audio/flac", "audio/aac", "audio/ogg", "audio/webm", "audio/m4a"
}

# 50 MB = 50,000 KB
MAX_SIZE_MB = 50
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024


@router.post("/", response_model=SmartTaskResponse)
async def suggest_tasks_endpoint(audio: UploadFile = File(...)):
    filename = audio.filename or "call_recording.wav"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    if audio.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=400, detail=f"Invalid MIME type: {audio.content_type}")

    content = await audio.read()
    if len(content) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed: {MAX_SIZE_MB}MB (50,000 KB). "
                   f"Current size: {len(content)/1024/1024:.1f}MB"
        )

    file_id = uuid.uuid4()
    temp_path = UPLOAD_DIR / f"{file_id}{ext}"

    try:
        temp_path.write_bytes(content)
        result_dict = suggest_next_tasks(str(temp_path), audio.content_type)
        return SmartTaskResponse(**result_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass