"""Property-based tests for backend service helpers."""

import torch
import pytest
from hypothesis import given, strategies as st

from backend.app.services.checkpoints import _extract_iter
from backend.app.services.inference import get_attention_map, get_layer_norms


@pytest.mark.property
@given(st.integers(min_value=0, max_value=1_000_000))
def test_extract_iter_round_trip(iter_num: int):
    filename = f"checkpoint_{iter_num}.pt"
    assert _extract_iter(filename) == iter_num


@pytest.mark.property
@given(
    layers=st.integers(min_value=1, max_value=3),
    heads=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=1, max_value=6),
)
def test_attention_map_shape_property(layers: int, heads: int, seq_len: int):
    patterns = [torch.rand(1, heads, seq_len, seq_len) for _ in range(layers)]
    diag = {"attention_patterns": patterns}

    attn = get_attention_map(diag, layer=layers - 1, head=heads - 1)
    assert len(attn) == seq_len
    assert all(len(row) == seq_len for row in attn)


@pytest.mark.property
@given(
    layers=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=1, max_value=6),
    d_model=st.integers(min_value=1, max_value=8),
)
def test_layer_norms_property(layers: int, seq_len: int, d_model: int):
    diag = {
        "layer_outputs": [torch.randn(1, seq_len, d_model) for _ in range(layers)]
    }
    norms = get_layer_norms(diag)
    assert len(norms) == layers
    for entry in norms:
        assert entry["layer"] >= 0
        assert isinstance(entry["avg_norm"], float)
