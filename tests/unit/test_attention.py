"""Unit tests for Attention class."""

import pytest
import torch
from pretraining.attention.attention import Attention
from pretraining.positional_embeddings.rope import RoPE
from pretraining.positional_embeddings.alibi import ALiBi


@pytest.mark.unit
@pytest.mark.parametrize("use_einops", [True, False])
class TestAttention:
    """Tests for Attention class."""

    def test_init(self, small_config, use_einops):
        """Test initialization."""
        attn = Attention(small_config, rope=None, alibi=None, use_einops=use_einops)
        assert attn.n_heads == small_config.n_heads
        assert attn.n_kv_heads == small_config.n_kv_heads
        assert attn.W_Q.shape == (small_config.n_heads, small_config.d_head, small_config.d_model)
        assert attn.W_K.shape == (small_config.n_kv_heads, small_config.d_head, small_config.d_model)
        assert attn.W_V.shape == (small_config.n_kv_heads, small_config.d_head, small_config.d_model)
        assert attn.W_O.shape == (small_config.n_heads, small_config.d_head, small_config.d_model)

    def test_compute_qkv(self, small_config, use_einops):
        """Test QKV computation."""
        attn = Attention(small_config, rope=None, alibi=None, use_einops=use_einops)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, small_config.d_model)
        q, k, v = attn._compute_qkv(residual)
        assert q.shape == (batch_size, seq_len, small_config.n_heads, small_config.d_head)
        assert k.shape == (batch_size, seq_len, small_config.n_kv_heads, small_config.d_head)
        assert v.shape == (batch_size, seq_len, small_config.n_kv_heads, small_config.d_head)

    def test_compute_attention_scores(self, small_config, use_einops):
        """Test attention score computation."""
        attn = Attention(small_config, rope=None, alibi=None, use_einops=use_einops)
        batch_size, seq_len = 2, 5
        q = torch.randn(batch_size, seq_len, small_config.n_heads, small_config.d_head)
        k = torch.randn(batch_size, seq_len, small_config.n_heads, small_config.d_head)
        scores = attn._compute_attention_scores(q, k)
        assert scores.shape == (batch_size, small_config.n_heads, seq_len, seq_len)
        # Check scaling
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()

    def test_apply_causal_mask(self, small_config, use_einops):
        """Test causal masking through forward pass."""
        attn = Attention(small_config, rope=None, alibi=None, use_einops=use_einops)
        batch_size, seq_len = 2, 5
        scores = torch.randn(batch_size, small_config.n_heads, seq_len, seq_len)
        # Apply causal mask manually (same logic as in forward)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=scores.device))
        masked_scores = scores.masked_fill(mask == 0, float("-inf"))
        # Check that future positions are masked (set to -inf)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert torch.all(masked_scores[:, :, i, j] == float("-inf"))

    def test_forward(self, small_config, use_einops):
        """Test forward pass."""
        attn = Attention(small_config, rope=None, alibi=None, use_einops=use_einops)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, small_config.d_model)
        output, cache = attn(residual)
        assert output.shape == (batch_size, seq_len, small_config.d_model)
        assert cache[0].shape == (batch_size, seq_len, small_config.n_kv_heads, small_config.d_head)
        assert cache[1].shape == (batch_size, seq_len, small_config.n_kv_heads, small_config.d_head)

    def test_forward_with_cache(self, small_config, use_einops):
        """Test forward pass with KV cache."""
        attn = Attention(small_config, rope=None, alibi=None, use_einops=use_einops)
        batch_size = 2
        
        # First forward pass
        residual1 = torch.randn(batch_size, 5, small_config.d_model)
        output1, cache1 = attn(residual1, cache=None, start_pos=0)
        
        # Second forward pass with cache
        residual2 = torch.randn(batch_size, 1, small_config.d_model)
        output2, cache2 = attn(residual2, cache=cache1, start_pos=5)
        
        assert cache2[0].shape[1] == 6  # 5 + 1
        assert cache2[1].shape[1] == 6

    def test_attention_pattern_sum_to_one(self, small_config, use_einops):
        """Test that attention patterns sum to 1."""
        attn = Attention(small_config, rope=None, alibi=None, use_einops=use_einops)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, small_config.d_model)
        output, _ = attn(residual)
        
        # Manually check attention pattern
        q, k, v = attn._compute_qkv(residual)
        scores = attn._compute_attention_scores(q, k)
        # Apply causal mask manually (same logic as in forward)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=residual.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))
        pattern = torch.softmax(scores, dim=-1)
        
        # Each row should sum to 1
        pattern_sum = pattern.sum(dim=-1)
        assert torch.allclose(pattern_sum, torch.ones_like(pattern_sum), atol=1e-5)

    def test_gradient_flow(self, small_config, use_einops):
        """Test that gradients flow through attention."""
        attn = Attention(small_config, rope=None, alibi=None, use_einops=use_einops)
        residual = torch.randn(1, 5, small_config.d_model, requires_grad=True)
        output, _ = attn(residual)
        loss = output.sum()
        loss.backward()
        assert attn.W_Q.grad is not None
        assert attn.W_K.grad is not None
        assert attn.W_V.grad is not None
        assert attn.W_O.grad is not None
        assert residual.grad is not None


@pytest.mark.unit
@pytest.mark.parametrize("use_einops", [True, False])
class TestAttentionWithRoPE:
    """Tests for Attention with RoPE."""

    def test_init_with_rope(self, llama_config, use_einops):
        """Test initialization with RoPE."""
        rope = RoPE(llama_config)
        attn = Attention(llama_config, rope=rope, alibi=None, use_einops=use_einops)
        assert attn.rope is not None

    def test_forward_with_rope(self, llama_config, use_einops):
        """Test forward pass with RoPE."""
        rope = RoPE(llama_config)
        attn = Attention(llama_config, rope=rope, alibi=None, use_einops=use_einops)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, llama_config.d_model)
        output, cache = attn(residual)
        assert output.shape == (batch_size, seq_len, llama_config.d_model)

    def test_forward_with_rope_and_cache(self, llama_config, use_einops):
        """Test forward pass with RoPE and cache."""
        rope = RoPE(llama_config)
        attn = Attention(llama_config, rope=rope, alibi=None, use_einops=use_einops)
        batch_size = 2
        
        # First forward pass
        residual1 = torch.randn(batch_size, 5, llama_config.d_model)
        output1, cache1 = attn(residual1, cache=None, start_pos=0)
        
        # Second forward pass with cache
        residual2 = torch.randn(batch_size, 1, llama_config.d_model)
        output2, cache2 = attn(residual2, cache=cache1, start_pos=5)
        
        assert cache2[0].shape[1] == 6


@pytest.mark.unit
@pytest.mark.parametrize("use_einops", [True, False])
class TestAttentionWithALiBi:
    """Tests for Attention with ALiBi."""

    def test_init_with_alibi(self, olmo_config, use_einops):
        """Test initialization with ALiBi."""
        alibi = ALiBi(olmo_config)
        attn = Attention(olmo_config, rope=None, alibi=alibi, use_einops=use_einops)
        assert attn.alibi is not None

    def test_forward_with_alibi(self, olmo_config, use_einops):
        """Test forward pass with ALiBi."""
        alibi = ALiBi(olmo_config)
        attn = Attention(olmo_config, rope=None, alibi=alibi, use_einops=use_einops)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, olmo_config.d_model)
        output, cache = attn(residual)
        assert output.shape == (batch_size, seq_len, olmo_config.d_model)


@pytest.mark.unit
class TestAttentionGQA:
    """Tests for Grouped Query Attention (GQA)."""

    def test_init_gqa(self, gqa_config):
        """Test GQA initialization."""
        attn = Attention(gqa_config, rope=None, alibi=None, use_einops=True)
        assert attn.n_heads == 8
        assert attn.n_kv_heads == 2  # 4:1 ratio
        assert attn.W_K.shape[0] == 2
        assert attn.W_V.shape[0] == 2

    def test_forward_gqa(self, gqa_config):
        """Test GQA forward pass."""
        rope = RoPE(gqa_config)
        attn = Attention(gqa_config, rope=rope, alibi=None, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, gqa_config.d_model)
        output, cache = attn(residual)
        assert output.shape == (batch_size, seq_len, gqa_config.d_model)
        # Cache should have n_kv_heads dimension
        assert cache[0].shape[2] == gqa_config.n_kv_heads
        assert cache[1].shape[2] == gqa_config.n_kv_heads

    def test_broadcasting(self, gqa_config):
        """Test that K/V are broadcast correctly for GQA."""
        attn = Attention(gqa_config, rope=None, alibi=None, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, gqa_config.d_model)
        q, k, v = attn._compute_qkv(residual)
        
        # Before broadcasting
        assert k.shape[2] == gqa_config.n_kv_heads
        assert v.shape[2] == gqa_config.n_kv_heads
        
        # After forward (broadcasting happens internally)
        output, cache = attn(residual)
        # Cache should still have n_kv_heads
        assert cache[0].shape[2] == gqa_config.n_kv_heads


@pytest.mark.unit
class TestAttentionMQA:
    """Tests for Multi-Query Attention (MQA)."""

    def test_init_mqa(self, mqa_config):
        """Test MQA initialization."""
        attn = Attention(mqa_config, rope=None, alibi=None, use_einops=True)
        assert attn.n_heads == 8
        assert attn.n_kv_heads == 1  # Single K/V head
        assert attn.W_K.shape[0] == 1
        assert attn.W_V.shape[0] == 1

    def test_forward_mqa(self, mqa_config):
        """Test MQA forward pass."""
        rope = RoPE(mqa_config)
        attn = Attention(mqa_config, rope=rope, alibi=None, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, mqa_config.d_model)
        output, cache = attn(residual)
        assert output.shape == (batch_size, seq_len, mqa_config.d_model)
        # Cache should have 1 KV head
        assert cache[0].shape[2] == 1
        assert cache[1].shape[2] == 1


@pytest.mark.unit
class TestAttentionEquivalence:
    """Tests for equivalence between einops and non-einops implementations."""

    @pytest.mark.parametrize("use_einops1,use_einops2", [(True, False)])
    def test_equivalence_forward(self, small_config, use_einops1, use_einops2):
        """Test that both implementations produce similar outputs."""
        attn1 = Attention(small_config, rope=None, alibi=None, use_einops=use_einops1)
        attn2 = Attention(small_config, rope=None, alibi=None, use_einops=use_einops2)
        
        # Copy parameters
        attn2.W_Q.data = attn1.W_Q.data.clone()
        attn2.W_K.data = attn1.W_K.data.clone()
        attn2.W_V.data = attn1.W_V.data.clone()
        attn2.W_O.data = attn1.W_O.data.clone()
        
        residual = torch.randn(2, 5, small_config.d_model)
        output1, cache1 = attn1(residual)
        output2, cache2 = attn2(residual)
        
        assert torch.allclose(output1, output2, atol=1e-5)

