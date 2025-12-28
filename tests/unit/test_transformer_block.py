"""Unit tests for TransformerBlock class."""

import pytest
import torch
from pretraining.transformer_blocks.transformer_block import TransformerBlock, create_transformer_block
from pretraining.positional_embeddings.rope import RoPE
from pretraining.positional_embeddings.alibi import ALiBi


@pytest.mark.unit
@pytest.mark.parametrize("use_einops", [True, False])
class TestTransformerBlock:
    """Tests for TransformerBlock."""

    def test_init(self, small_config, use_einops):
        """Test initialization."""
        block = TransformerBlock(small_config, rope=None, alibi=None, use_einops=use_einops)
        assert hasattr(block, 'ln1')
        assert hasattr(block, 'attn')
        assert hasattr(block, 'ln2')
        assert hasattr(block, 'mlp')

    def test_forward(self, small_config, use_einops):
        """Test forward pass."""
        block = TransformerBlock(small_config, rope=None, alibi=None, use_einops=use_einops)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, small_config.d_model)
        output, cache, aux_loss = block(residual)
        assert output.shape == residual.shape
        assert cache[0].shape == (batch_size, seq_len, small_config.n_kv_heads, small_config.d_head)
        assert aux_loss is None  # No MoE

    def test_forward_with_cache(self, small_config, use_einops):
        """Test forward pass with KV cache."""
        block = TransformerBlock(small_config, rope=None, alibi=None, use_einops=use_einops)
        batch_size = 2
        
        # First forward pass
        residual1 = torch.randn(batch_size, 5, small_config.d_model)
        output1, cache1, aux_loss1 = block(residual1, cache=None, start_pos=0)
        
        # Second forward pass with cache
        residual2 = torch.randn(batch_size, 1, small_config.d_model)
        output2, cache2, aux_loss2 = block(residual2, cache=cache1, start_pos=5)
        
        assert cache2[0].shape[1] == 6  # 5 + 1

    def test_residual_connections(self, small_config, use_einops):
        """Test that residual connections work."""
        block = TransformerBlock(small_config, rope=None, alibi=None, use_einops=use_einops)
        residual = torch.randn(1, 5, small_config.d_model)
        output, _, _ = block(residual)
        
        # Output should be different from input (transformation applied)
        assert not torch.allclose(output, residual, atol=1e-5)

    def test_gradient_flow(self, small_config, use_einops):
        """Test that gradients flow through transformer block."""
        block = TransformerBlock(small_config, rope=None, alibi=None, use_einops=use_einops)
        residual = torch.randn(1, 5, small_config.d_model, requires_grad=True)
        output, _, _ = block(residual)
        loss = output.sum()
        loss.backward()
        assert residual.grad is not None


@pytest.mark.unit
class TestTransformerBlockWithRoPE:
    """Tests for TransformerBlock with RoPE."""

    def test_init_with_rope(self, llama_config):
        """Test initialization with RoPE."""
        rope = RoPE(llama_config)
        block = TransformerBlock(llama_config, rope=rope, alibi=None, use_einops=True)
        assert block.attn.rope is not None

    def test_forward_with_rope(self, llama_config):
        """Test forward pass with RoPE."""
        rope = RoPE(llama_config)
        block = TransformerBlock(llama_config, rope=rope, alibi=None, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, llama_config.d_model)
        output, cache, aux_loss = block(residual)
        assert output.shape == residual.shape


@pytest.mark.unit
class TestTransformerBlockWithALiBi:
    """Tests for TransformerBlock with ALiBi."""

    def test_init_with_alibi(self, olmo_config):
        """Test initialization with ALiBi."""
        alibi = ALiBi(olmo_config)
        block = TransformerBlock(olmo_config, rope=None, alibi=alibi, use_einops=True)
        assert block.attn.alibi is not None

    def test_forward_with_alibi(self, olmo_config):
        """Test forward pass with ALiBi."""
        alibi = ALiBi(olmo_config)
        block = TransformerBlock(olmo_config, rope=None, alibi=alibi, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, olmo_config.d_model)
        output, cache, aux_loss = block(residual)
        assert output.shape == residual.shape


@pytest.mark.unit
class TestTransformerBlockWithMoE:
    """Tests for TransformerBlock with MoE."""

    def test_forward_with_moe(self, moe_config):
        """Test forward pass with MoE."""
        rope = RoPE(moe_config)
        block = TransformerBlock(moe_config, rope=rope, alibi=None, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, moe_config.d_model)
        block.train()  # Enable training to get aux_loss
        output, cache, aux_loss = block(residual)
        assert output.shape == residual.shape
        assert aux_loss is not None  # MoE produces aux_loss


@pytest.mark.unit
class TestCreateTransformerBlock:
    """Tests for create_transformer_block factory function."""

    def test_create_block(self, small_config):
        """Test creating transformer block."""
        block = create_transformer_block(small_config, use_einops=True)
        assert isinstance(block, TransformerBlock)

    def test_create_block_with_rope(self, llama_config):
        """Test creating block with RoPE."""
        rope = RoPE(llama_config)
        block = create_transformer_block(llama_config, use_einops=True, rope=rope)
        assert block.attn.rope is not None

    def test_create_block_with_alibi(self, olmo_config):
        """Test creating block with ALiBi."""
        alibi = ALiBi(olmo_config)
        block = create_transformer_block(olmo_config, use_einops=True, alibi=alibi)
        assert block.attn.alibi is not None

