"""Checkpoint discovery utilities for API responses."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def list_checkpoints(base_dir: str = "checkpoints") -> List[Dict[str, object]]:
    checkpoints_dir = Path(base_dir)
    if not checkpoints_dir.exists():
        return []

    checkpoints: List[Dict[str, object]] = []
    for checkpoint_dir in sorted(checkpoints_dir.iterdir(), reverse=True):
        if not checkpoint_dir.is_dir():
            continue

        # Pre-trained checkpoints
        final_model = checkpoint_dir / "final_model.pt"
        if final_model.exists():
            checkpoints.append(_make_checkpoint_entry(final_model, checkpoint_dir.name, False))
        else:
            for ckpt_file in sorted(checkpoint_dir.glob("checkpoint_*.pt"), reverse=True):
                checkpoints.append(_make_checkpoint_entry(ckpt_file, checkpoint_dir.name, False))

        # Fine-tuned checkpoints
        sft_dir = checkpoint_dir / "sft"
        if sft_dir.exists():
            sft_final = sft_dir / "final_model.pt"
            if sft_final.exists():
                checkpoints.append(_make_checkpoint_entry(sft_final, checkpoint_dir.name, True))
            else:
                for ckpt_file in sorted(sft_dir.glob("checkpoint_*.pt"), reverse=True):
                    checkpoints.append(_make_checkpoint_entry(ckpt_file, checkpoint_dir.name, True))

    return checkpoints


def _make_checkpoint_entry(path: Path, run_id: str, is_finetuned: bool) -> Dict[str, object]:
    iter_num = _extract_iter(path.name)
    return {
        "id": str(path),
        "path": str(path),
        "run_id": run_id,
        "is_finetuned": is_finetuned,
        "name": path.name,
        "iter": iter_num,
        "mtime": path.stat().st_mtime,
        "size_bytes": path.stat().st_size,
    }


def _extract_iter(filename: str) -> int | None:
    if filename.startswith("checkpoint_") and filename.endswith(".pt"):
        try:
            return int(filename.split("checkpoint_")[1].split(".pt")[0])
        except ValueError:
            return None
    return None


def resolve_checkpoint_path(checkpoint_id: str) -> Path:
    path = Path(checkpoint_id).resolve()
    base = Path("checkpoints").resolve()
    if not str(path).startswith(str(base)):
        raise ValueError("Checkpoint path must be within checkpoints directory")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path
