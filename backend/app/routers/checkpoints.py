"""Checkpoint listing endpoints."""

from __future__ import annotations

from typing import Any, Dict

import torch
from fastapi import APIRouter, HTTPException

from backend.app.services.checkpoints import list_checkpoints, resolve_checkpoint_path
from config import ModelConfig
from pretraining.training.training_args import TransformerTrainingArgs

try:
    from finetuning.training.finetuning_args import FinetuningArgs
except ImportError:  # pragma: no cover - optional dependency
    FinetuningArgs = None

router = APIRouter(prefix="/api/checkpoints")


@router.get("")
async def checkpoints() -> dict:
    return {"checkpoints": list_checkpoints()}


@router.get("/{checkpoint_id:path}")
async def checkpoint_info(checkpoint_id: str) -> dict:
    try:
        path = resolve_checkpoint_path(checkpoint_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    safe_globals = [TransformerTrainingArgs]
    if FinetuningArgs is not None:
        safe_globals.append(FinetuningArgs)
    torch.serialization.add_safe_globals(safe_globals)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("cfg")
    cfg_dict: Dict[str, Any]
    if isinstance(cfg, ModelConfig):
        cfg_dict = cfg.to_dict()
    elif isinstance(cfg, dict):
        cfg_dict = ModelConfig.from_dict(cfg).to_dict()
    elif cfg is None:
        cfg_dict = ModelConfig.gpt_small().to_dict()
    else:
        cfg_dict = ModelConfig.gpt_small().to_dict()

    return {
        "path": str(path),
        "tokenizer_type": checkpoint.get("tokenizer_type"),
        "cfg": cfg_dict,
        "iter_num": checkpoint.get("iter_num"),
        "is_finetuned": checkpoint.get("is_finetuned", False),
        "lora_info": checkpoint.get("lora_info"),
        "use_lora": checkpoint.get("use_lora", False),
    }
