"""Pydantic schemas for inference endpoints."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class InferenceSessionRequest(BaseModel):
    checkpoint_id: str


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.9


class DiagnosticsRequest(BaseModel):
    prompt: str

