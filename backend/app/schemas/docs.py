"""Pydantic schemas for documentation endpoints."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field


class ModelCodeRequest(BaseModel):
    model_config_data: Dict[str, Any] = Field(alias="model_config")

    model_config = ConfigDict(populate_by_name=True)


class FinetuningCodeRequest(BaseModel):
    use_lora: bool = False

    model_config = ConfigDict(populate_by_name=True)
