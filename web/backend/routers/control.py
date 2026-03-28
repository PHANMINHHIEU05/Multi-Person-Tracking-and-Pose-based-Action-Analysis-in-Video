"""POST /api/start – khởi chạy pipeline
   POST /api/stop/{run_id} – dừng pipeline
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse  # REFACTOR: serve uploaded video via HTTP range requests
from pydantic import BaseModel

from web.backend.core.pipeline import start_pipeline_thread
from web.backend.core.session import create_session, get_session
from web.backend.db.database import insert_run, update_run

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = str(
    Path(__file__).parent.parent.parent.parent
    / "runs/train_horizontal/final_safe_system.pth"
)

router = APIRouter()


def get_upload_path(file_id: str) -> Path:  # REFACTOR: shared resolver for uploaded input video
    matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))  # REFACTOR: map file_id to actual uploaded file path
    if not matches:  # REFACTOR: explicit not-found branch for stable API errors
        raise HTTPException(404, f"file_id '{file_id}' not found in uploads")  # REFACTOR: preserve existing error semantics
    return matches[0]  # REFACTOR: return resolved upload path for start + direct streaming endpoint


class StartRequest(BaseModel):
    file_id: str
    model_path: str = DEFAULT_MODEL
    conf: float = 0.25
    imgsz: int = 640


@router.post("/start")
async def start_pipeline(body: StartRequest, request: Request):
    # Tìm file đã upload
    upload_path = get_upload_path(body.file_id)  # REFACTOR: centralize file resolution logic
    video_path = str(upload_path)  # REFACTOR: pipeline consumes absolute path string

    run_id = uuid.uuid4().hex[:12]
    out_dir = OUTPUT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    session = create_session(run_id, video_path, str(out_dir))
    await insert_run(run_id, upload_path.name, str(out_dir))  # REFACTOR: keep DB history linked to original filename

    # Lấy running event loop để thread có thể push vào queue
    loop = asyncio.get_event_loop()
    start_pipeline_thread(session, body.model_path, body.conf, body.imgsz, loop)

    return {"run_id": run_id, "ws_url": f"/ws/{run_id}"}


@router.get("/video/{file_id}")
async def serve_video(file_id: str):
    path = get_upload_path(file_id)  # REFACTOR: stream original uploaded file directly to browser
    return FileResponse(  # REFACTOR: byte-range capable response for native HTML5 video playback/seek
        str(path),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )


@router.post("/stop/{run_id}")
async def stop_pipeline(run_id: str):
    session = get_session(run_id)
    if session is None:
        raise HTTPException(404, f"run_id '{run_id}' not found")
    session.stop_event.set()
    await update_run(run_id, "stopped",
                     total_frames=session.total_frames,
                     fall_count=session.fall_count)
    return {"run_id": run_id, "status": "stopped"}
