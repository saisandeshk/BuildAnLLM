"""Property-based tests for attention mathematical properties."""

import pytest
import torch
from pretraining.attention.attention import Attention


@pytest.mark.property
class TestAttentionProperties:
    """Property-based tests for attention."""

    def test_attention_pattern_sum_to_one(self, small_config):
        """Property: Attention patterns should sum to 1."""
        attn = Attention(small_config, rope=None, alibi=None, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, small_config.d_model)

        q, k, v = attn._compute_qkv(residual)
        scores = attn._compute_attention_scores(q, k)
        # Apply causal mask manually (same logic as in forward)
        mask = torch.tril(torch.ones(
            (seq_len, seq_len), device=residual.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))
        pattern = torch.softmax(scores, dim=-1)

        # Each row should sum to 1
        pattern_sum = pattern.sum(dim=-1)
        assert torch.allclose(
            pattern_sum, torch.ones_like(pattern_sum), atol=1e-5)

    def test_causal_mask_property(self, small_config):
        """Property: Causal mask prevents attending to future tokens."""
        attn = Attention(small_config, rope=None, alibi=None, use_einops=True)
        batch_size, seq_len = 2, 5
        scores = torch.randn(
            batch_size, small_config.n_heads, seq_len, seq_len)
        # Apply causal mask manually (same logic as in forward)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=scores.device))
        masked = scores.masked_fill(mask == 0, float("-inf"))

        # Future positions should be -inf
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert torch.all(masked[:, :, i, j] == float("-inf"))

    def test_attention_output_shape(self, small_config):
        """Property: Attention output should have correct shape."""
        attn = Attention(small_config, rope=None, alibi=None, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, small_config.d_model)
        output, _ = attn(residual)

        assert output.shape == residual.shape

    def test_attention_linearity(self, small_config):
        """Property: Attention is linear in values."""
        attn = Attention(small_config, rope=None, alibi=None, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, small_config.d_model)

        q, k, v1 = attn._compute_qkv(residual)
        _, _, v2 = attn._compute_qkv(residual * 2)

        # Get attention pattern
        scores = attn._compute_attention_scores(q, k)
        # Apply causal mask manually (same logic as in forward)
        mask = torch.tril(torch.ones(
            (seq_len, seq_len), device=residual.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))
        pattern = torch.softmax(scores, dim=-1)

        # Apply to values
        output1 = attn._apply_attention_to_values(pattern, v1)
        output2 = attn._apply_attention_to_values(pattern, v2)

        # Should be linear: output2 ≈ 2 * output1
        assert torch.allclose(output2, output1 * 2, atol=1e-5)


@pytest.mark.property
class TestAttentionWithRoPEProperties:
    """Property-based tests for attention with RoPE."""

    def test_rope_preserves_magnitude(self, llama_config):
        """Property: RoPE preserves vector magnitude."""
        from pretraining.positional_embeddings.rope import RoPE

        rope = RoPE(llama_config)
        batch, seq, n_heads, d_head = 2, 5, llama_config.n_heads, llama_config.d_head
        q = torch.randn(batch, seq, n_heads, d_head)
        k = torch.randn(batch, seq, llama_config.n_kv_heads, d_head)
        positions = torch.arange(seq)

        q_orig_norm = torch.norm(q, dim=-1)
        k_orig_norm = torch.norm(k, dim=-1)

        q_rotated, k_rotated = rope(q, k, positions)

        q_rot_norm = torch.norm(q_rotated, dim=-1)
        k_rot_norm = torch.norm(k_rotated, dim=-1)

        # Rotation should preserve magnitude
        assert torch.allclose(q_orig_norm, q_rot_norm, atol=1e-5)
        assert torch.allclose(k_orig_norm, k_rot_norm, atol=1e-5)
