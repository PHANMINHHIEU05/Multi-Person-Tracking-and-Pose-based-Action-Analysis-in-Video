"""GET /api/history – danh sách lịch sử các lần chạy"""
from fastapi import APIRouter
from web.backend.db.database import get_history

router = APIRouter()


@router.get("/history")
async def history():
    return await get_history()
