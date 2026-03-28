"""Session manager – in-memory state per run."""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RunSession:
    run_id: str
    video_path: str
    output_dir: str
    status: str = "pending"          # pending | running | done | stopped | error
    # Keep this queue small to prioritize low-latency preview over completeness.
    # If the browser is slower than inference, old frames are dropped quickly.
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=12))
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    total_frames: int = 0
    fall_count: int = 0
    error_msg: str = ""


_sessions: dict[str, RunSession] = {}


def create_session(run_id: str, video_path: str, output_dir: str) -> RunSession:
    s = RunSession(run_id=run_id, video_path=video_path, output_dir=output_dir)
    _sessions[run_id] = s
    return s


def get_session(run_id: str) -> Optional[RunSession]:
    return _sessions.get(run_id)


def all_sessions() -> dict[str, RunSession]:
    return _sessions
