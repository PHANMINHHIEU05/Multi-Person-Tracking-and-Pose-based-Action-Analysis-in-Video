"""POST /api/start – khởi chạy pipeline
   POST /api/stop/{run_id} – dừng pipeline
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from web.backend.core.pipeline import start_pipeline_thread
from web.backend.core.session import create_session, get_session
from web.backend.db.database import insert_run, update_run

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = str(
    Path(__file__).parent.parent.parent.parent
    / "runs/train_v3/final_safe_system.pth"
)

router = APIRouter()


class StartRequest(BaseModel):
    file_id: str
    model_path: str = DEFAULT_MODEL
    conf: float = 0.25
    imgsz: int = 640


@router.post("/start")
async def start_pipeline(body: StartRequest, request: Request):
    # Tìm file đã upload
    matches = list(UPLOAD_DIR.glob(f"{body.file_id}.*"))
    if not matches:
        raise HTTPException(404, f"file_id '{body.file_id}' not found in uploads")
    video_path = str(matches[0])

    run_id = uuid.uuid4().hex[:12]
    out_dir = OUTPUT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    session = create_session(run_id, video_path, str(out_dir))
    await insert_run(run_id, matches[0].name, str(out_dir))

    # Lấy running event loop để thread có thể push vào queue
    loop = asyncio.get_event_loop()
    start_pipeline_thread(session, body.model_path, body.conf, body.imgsz, loop)

    return {"run_id": run_id, "ws_url": f"/ws/{run_id}"}


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
