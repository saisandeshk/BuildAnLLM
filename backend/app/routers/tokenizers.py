"""Tokenizer utility endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import tiktoken

router = APIRouter(prefix="/api/tokenizers")


class TiktokenEncodeRequest(BaseModel):
    model: str
    text: str


@router.get("/tiktoken/models")
async def tiktoken_models() -> dict:
    models = list(tiktoken.model.MODEL_TO_ENCODING.keys())
    return {"models": sorted(models)}


@router.post("/tiktoken/encode")
async def tiktoken_encode(request: TiktokenEncodeRequest) -> dict:
    try:
        encoding = tiktoken.encoding_for_model(request.model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(request.text)
    decoded_tokens = [encoding.decode([tok]) for tok in tokens]

    return {
        "tokens": tokens,
        "decoded_tokens": decoded_tokens,
        "token_count": len(tokens),
        "char_count": len(request.text),
        "chars_per_token": (len(request.text) / len(tokens)) if tokens else 0,
        "model": request.model,
    }

