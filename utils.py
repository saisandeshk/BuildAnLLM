"""Shared utility functions for training and inference."""

import torch
import time
import streamlit as st
from typing import List
from pathlib import Path


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_state_dict_warnings(unexpected_keys: List[str], missing_keys: List[str]) -> None:
    """Print warnings about state dict mismatches."""
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected key(s) in checkpoint (ignored):")
        for key in unexpected_keys[:5]:
            print(f"  - {key}")
        if len(unexpected_keys) > 5:
            print(f"  ... and {len(unexpected_keys) - 5} more")

    if missing_keys:
        print(f"Warning: {len(missing_keys)} missing key(s) in checkpoint (using random initialization):")
        for key in missing_keys[:5]:
            print(f"  - {key}")
        if len(missing_keys) > 5:
            print(f"  ... and {len(missing_keys) - 5} more")


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in a human-readable format.

    Args:
        seconds: Elapsed time in seconds

    Returns:
        Formatted string (e.g., "45.2s", "5m 30s", "2h 15m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def get_elapsed_time() -> float:
    """Get current elapsed training time from session state.

    Returns:
        Elapsed time in seconds, or 0.0 if training hasn't started
    """
    if "training_start_time" in st.session_state:
        return time.time() - st.session_state.training_start_time
    return 0.0


def get_total_training_time() -> float:
    """Get total training time from session state.

    Uses training_end_time if available, otherwise calculates from current time.

    Returns:
        Total time in seconds, or 0.0 if training hasn't started or if time is invalid
    """
    if "training_start_time" not in st.session_state:
        return 0.0

    if "training_end_time" in st.session_state:
        elapsed = st.session_state.training_end_time - st.session_state.training_start_time
    else:
        elapsed = time.time() - st.session_state.training_start_time
    
    # Return 0.0 if time is negative or invalid (prevents display of negative times)
    return max(0.0, elapsed)


def scan_checkpoints():
    """Scan checkpoints directory and return available checkpoints."""
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        return []

    checkpoints = []
    for checkpoint_dir in sorted(checkpoints_dir.iterdir(), reverse=True):
        if checkpoint_dir.is_dir():
            # Check for pre-trained checkpoints
            final_model = checkpoint_dir / "final_model.pt"
            if final_model.exists():
                checkpoints.append({
                    "path": str(final_model),
                    "name": f"{checkpoint_dir.name} (final)",
                    "timestamp": checkpoint_dir.name,
                    "is_finetuned": False
                })
            else:
                # Get all checkpoint files
                for ckpt_file in sorted(checkpoint_dir.glob("checkpoint_*.pt"), reverse=True):
                    checkpoints.append({
                        "path": str(ckpt_file),
                        "name": f"{checkpoint_dir.name} / {ckpt_file.stem}",
                        "timestamp": checkpoint_dir.name,
                        "is_finetuned": False
                    })
            
            # Check for fine-tuned checkpoints in sft/ subdirectory
            sft_dir = checkpoint_dir / "sft"
            if sft_dir.exists():
                sft_final = sft_dir / "final_model.pt"
                if sft_final.exists():
                    checkpoints.append({
                        "path": str(sft_final),
                        "name": f"{checkpoint_dir.name} / sft (final)",
                        "timestamp": checkpoint_dir.name,
                        "is_finetuned": True
                    })
                else:
                    # Get all SFT checkpoint files
                    for ckpt_file in sorted(sft_dir.glob("checkpoint_*.pt"), reverse=True):
                        checkpoints.append({
                            "path": str(ckpt_file),
                            "name": f"{checkpoint_dir.name} / sft / {ckpt_file.stem}",
                            "timestamp": checkpoint_dir.name,
                            "is_finetuned": True
                        })

    return checkpoints

