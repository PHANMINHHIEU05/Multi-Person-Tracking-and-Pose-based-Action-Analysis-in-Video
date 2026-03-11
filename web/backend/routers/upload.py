"""POST /api/upload – nhận video file, lưu vào uploads/"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_MIME = {"video/mp4", "video/avi", "video/x-msvideo", "video/quicktime",
                "video/x-matroska", "video/webm"}

router = APIRouter()


@router.post("/upload")
async def upload_video(file: UploadFile):
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: mp4, avi, mov, mkv, webm",
        )

    file_id = uuid.uuid4().hex
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    dest = UPLOAD_DIR / f"{file_id}{suffix}"

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"file_id": file_id, "filename": file.filename, "path": str(dest)}
