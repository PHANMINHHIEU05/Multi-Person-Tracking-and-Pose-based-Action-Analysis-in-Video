"""WebSocket stream endpoint.

Protocol:
  Video frame → two WS messages:
    1) binary message: raw JPEG bytes
    2) text message:   JSON metadata {type:"frame", frame_idx, fps, tracks, action_counts, fall_alert}
  Status → one text message: JSON {type:"status", status, message}
  Heartbeat → text: {type:"ping"}
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from web.backend.core.session import get_session

router = APIRouter()


@router.websocket("/ws/{run_id}")
async def websocket_stream(websocket: WebSocket, run_id: str):
    await websocket.accept()
    session = get_session(run_id)
    if session is None:
        await websocket.send_text(json.dumps(
            {"type": "status", "status": "error", "message": f"run_id '{run_id}' not found"}
        ))
        await websocket.close()
        return

    try:
        while True:
            try:
                item = await asyncio.wait_for(session.queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
                continue

            jpeg_bytes, meta = item

            if jpeg_bytes is not None:
                # Frame: send binary JPEG first, then JSON metadata
                await websocket.send_bytes(jpeg_bytes)
                await websocket.send_text(json.dumps(meta))
            else:
                # Status message (no image)
                await websocket.send_text(json.dumps(meta))

            if meta.get("type") == "status" and meta.get("status") in ("done", "stopped", "error"):
                break

    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()
