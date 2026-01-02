"""Shared utility functions for training and inference."""

import torch
from typing import List


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


