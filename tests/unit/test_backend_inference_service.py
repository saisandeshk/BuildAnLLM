"""Unit tests for inference service helpers."""

import torch
import pytest

from backend.app.services.inference import get_attention_map, get_layer_norms, get_logit_lens


class DummyTokenizer:
    def decode(self, token_ids):
        return f"T{token_ids[0]}"


class DummyUnembed:
    def __init__(self, d_model: int, vocab_size: int) -> None:
        self.W_U = torch.randn(d_model, vocab_size)


class DummyModel:
    def __init__(self, d_model: int = 4, vocab_size: int = 10) -> None:
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.unembed = DummyUnembed(d_model, vocab_size)
        self.ln_f = torch.nn.Identity()

    def parameters(self):
        return iter([self._param])


@pytest.mark.unit
class TestInferenceServices:
    def test_get_attention_map_shape(self):
        seq_len = 4
        heads = 2
        attn = torch.rand(1, heads, seq_len, seq_len)
        diag = {"attention_patterns": [attn]}

        result = get_attention_map(diag, layer=0, head=1)
        assert len(result) == seq_len
        assert all(len(row) == seq_len for row in result)

    def test_get_layer_norms_shape(self):
        diag = {
            "layer_outputs": [
                torch.randn(1, 3, 4),
                torch.randn(1, 3, 4),
            ]
        }
        norms = get_layer_norms(diag)
        assert len(norms) == 2
        assert {entry["layer"] for entry in norms} == {0, 1}

    def test_get_logit_lens_returns_top_k(self):
        model = DummyModel(d_model=4, vocab_size=8)
        tokenizer = DummyTokenizer()
        diag = {
            "layer_outputs": [
                torch.randn(1, 2, 4),
                torch.randn(1, 2, 4),
            ]
        }
        result = get_logit_lens(diag, model, tokenizer, position=0, top_k=3)
        assert len(result) == 2
        for layer in result:
            assert len(layer["predictions"]) == 3
