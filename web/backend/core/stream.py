"""WebSocket stream endpoint."""
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
                await websocket.send_text(json.dumps({"type": "ping"}))  # REFACTOR: keep connection alive while inference is running
                continue

            await websocket.send_text(json.dumps(item))  # REFACTOR: JSON-only websocket transport (no binary frames)

            if item.get("type") == "status" and item.get("status") in ("done", "stopped", "error"):
                break

    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()  # REFACTOR: clean close after terminal status/disconnect
        except RuntimeError:
            pass  # Connection already closed by client
