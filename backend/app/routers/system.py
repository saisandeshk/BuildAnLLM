"""System and health endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.services.system_info import get_system_info

router = APIRouter(prefix="/api")


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/system/info")
async def system_info() -> dict:
    return get_system_info()

