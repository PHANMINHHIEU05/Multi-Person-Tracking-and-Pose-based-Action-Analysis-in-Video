"""Database layer – SQLite via aiosqlite."""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import aiosqlite

DB_PATH = Path(__file__).parent / "runs.db"


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id          TEXT PRIMARY KEY,
                video_name  TEXT NOT NULL,
                started_at  TEXT NOT NULL,
                finished_at TEXT,
                status      TEXT NOT NULL DEFAULT 'pending',
                output_dir  TEXT,
                total_frames INTEGER DEFAULT 0,
                fall_count  INTEGER DEFAULT 0
            )
        """)
        await db.commit()


async def insert_run(run_id: str, video_name: str, output_dir: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO runs (id, video_name, started_at, status, output_dir) VALUES (?,?,?,?,?)",
            (run_id, video_name, datetime.utcnow().isoformat(), "running", output_dir),
        )
        await db.commit()


async def update_run(run_id: str, status: str, total_frames: int = 0, fall_count: int = 0):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE runs SET status=?, finished_at=?, total_frames=?, fall_count=?
               WHERE id=?""",
            (status, datetime.utcnow().isoformat(), total_frames, fall_count, run_id),
        )
        await db.commit()


async def get_history() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT 50"
        ) as cursor:
            rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def get_run(run_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM runs WHERE id=?", (run_id,)) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None
