"""GET /api/results/{run_id} – tải video output và CSV"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from web.backend.core.session import get_session
from web.backend.db.database import get_run

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

router = APIRouter()


@router.get("/results/{run_id}/video")
async def download_video(run_id: str):
    run = await get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    path = Path(run["output_dir"]) / "video_action.mp4"
    if not path.exists():
        raise HTTPException(404, "Video not ready yet")
    return FileResponse(str(path), media_type="video/mp4",
                        filename=f"action_{run_id}.mp4")


@router.get("/results/{run_id}/csv")
async def download_csv(run_id: str):
    run = await get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    path = Path(run["output_dir"]) / "actions.csv"
    if not path.exists():
        raise HTTPException(404, "CSV not ready yet")
    return FileResponse(str(path), media_type="text/csv",
                        filename=f"actions_{run_id}.csv")


@router.get("/results/{run_id}")
async def get_result_meta(run_id: str):
    run = await get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    session = get_session(run_id)
    return {
        **run,
        "live_status": session.status if session else run["status"],
        "total_frames": session.total_frames if session else run["total_frames"],
        "fall_count": session.fall_count if session else run["fall_count"],
    }
