"""FastAPI entrypoint for the transformer backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routers import checkpoints, docs, finetune, inference, pretrain, system, tokenizers

app = FastAPI(title="Transformer Training API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(docs.router)
app.include_router(checkpoints.router)
app.include_router(pretrain.router)
app.include_router(finetune.router)
app.include_router(inference.router)
app.include_router(tokenizers.router)


@app.get("/")
async def root() -> dict:
    return {"service": "transformer-backend"}

