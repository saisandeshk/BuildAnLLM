"""Documentation endpoints (code snippets)."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.schemas.docs import FinetuningCodeRequest, ModelCodeRequest
from backend.app.services.code_snippets import (
    build_finetuning_code_snippets,
    build_inference_code_snippets,
    build_model_code_snippets,
)

router = APIRouter(prefix="/api/docs")


@router.post("/model-code")
async def model_code(request: ModelCodeRequest) -> dict:
    return {"snippets": build_model_code_snippets(request.model_config_data)}


@router.get("/inference-code")
async def inference_code() -> dict:
    return {"snippets": build_inference_code_snippets()}


@router.post("/finetuning-code")
async def finetuning_code(request: FinetuningCodeRequest) -> dict:
    return {"snippets": build_finetuning_code_snippets(request.use_lora)}
