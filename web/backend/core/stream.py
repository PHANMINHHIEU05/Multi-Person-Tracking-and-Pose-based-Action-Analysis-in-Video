"""WebSocket stream endpoint."""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from web.backend.core.session import get_session

router = APIRouter()

# PERF: Global preview queue keeps only the latest frame (drops stale frames).
_preview_queue: queue.Queue[bytes] = queue.Queue(maxsize=1)
# PERF: Cap preview sending cadence so browser playback does not run unnaturally fast.
PREVIEW_MAX_FPS = 24.0


def push_preview_frame(jpeg_bytes: bytes) -> None:
    """PERF: Put latest preview frame without blocking; drop old frame if full."""
    if not jpeg_bytes:
        return
    try:
        _preview_queue.put_nowait(jpeg_bytes)
    except queue.Full:
        try:
            _preview_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            _preview_queue.put_nowait(jpeg_bytes)
        except queue.Full:
            pass


async def ws_sender_loop(websocket: WebSocket, stop_evt: threading.Event) -> None:
    """PERF: Background sender loop that forwards latest JPEG bytes to websocket."""
    loop = asyncio.get_running_loop()
    interval = 1.0 / PREVIEW_MAX_FPS
    next_send_ts = time.perf_counter()
    while not stop_evt.is_set():
        try:
            jpeg_bytes = await loop.run_in_executor(None, _preview_queue.get)
        except Exception:
            continue
        if stop_evt.is_set():
            break
        if jpeg_bytes:
            # PERF: Pace outgoing frames to stable cadence for smoother playback.
            now = time.perf_counter()
            if now < next_send_ts:
                await asyncio.sleep(next_send_ts - now)
            await websocket.send_bytes(jpeg_bytes)
            now = time.perf_counter()
            if now > next_send_ts + interval:
                next_send_ts = now + interval
            else:
                next_send_ts += interval


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

    sender_stop = threading.Event()
    sender_task = asyncio.create_task(ws_sender_loop(websocket, sender_stop))

    try:
        while True:
            try:
                item = await asyncio.wait_for(session.queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
                continue

            # PERF: Metadata/status is text; preview bytes are sent by ws_sender_loop.
            await websocket.send_text(json.dumps(item))

            if item.get("type") == "status" and item.get("status") in ("done", "stopped", "error"):
                break

    except WebSocketDisconnect:
        pass
    finally:
        sender_stop.set()
        sender_task.cancel()
        await asyncio.gather(sender_task, return_exceptions=True)
        await websocket.close()
