"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from web.backend.db.database import init_db
from web.backend.routers import upload, control, results, history
from web.backend.core.stream import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Action Analysis Web UI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST API
app.include_router(upload.router,  prefix="/api")
app.include_router(control.router, prefix="/api")
app.include_router(results.router, prefix="/api")
app.include_router(history.router, prefix="/api")

# WebSocket
app.include_router(ws_router)

# Serve React build (sau khi npm run build)
STATIC_DIR = Path(__file__).parent.parent / "frontend" / "dist"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
