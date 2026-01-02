"""ModelConfig construction helpers for API requests."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict

from config import ModelConfig


def build_model_config(raw_config: Dict[str, Any]) -> ModelConfig:
    if not raw_config:
        raise ValueError("model_config is required")

    config = dict(raw_config)
    allowed = {field.name for field in fields(ModelConfig)}
    filtered = {key: value for key, value in config.items() if key in allowed}
    return ModelConfig.from_dict(filtered)
