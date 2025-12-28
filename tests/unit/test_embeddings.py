"""Unit tests for embedding classes."""

import pytest
import torch
from pretraining.embeddings.embed import (
    EmbedWithoutTorch,
    EmbedWithTorch,
    UnembedWithoutTorch,
    UnembedWithTorch,
)


@pytest.mark.unit
class TestEmbedWithoutTorch:
    """Tests for EmbedWithoutTorch."""

    def test_init(self, small_config):
        """Test initialization."""
        embed = EmbedWithoutTorch(small_config)
        assert embed.W_E.shape == (small_config.d_vocab, small_config.d_model)
        assert isinstance(embed.W_E, torch.nn.Parameter)

    def test_forward(self, small_config, sample_tokens):
        """Test forward pass."""
        embed = EmbedWithoutTorch(small_config)
        output = embed(sample_tokens)
        assert output.shape == (sample_tokens.shape[0], sample_tokens.shape[1], small_config.d_model)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_batch(self, small_config, sample_batch_tokens):
        """Test forward pass with batch."""
        embed = EmbedWithoutTorch(small_config)
        output = embed(sample_batch_tokens)
        assert output.shape == (sample_batch_tokens.shape[0], sample_batch_tokens.shape[1], small_config.d_model)

    def test_embedding_lookup(self, small_config):
        """Test that embeddings are correctly looked up."""
        embed = EmbedWithoutTorch(small_config)
        tokens = torch.tensor([[0, 1, 2]], dtype=torch.long)
        output = embed(tokens)
        # Check that output matches embedding matrix rows
        assert torch.allclose(output[0, 0], embed.W_E[0])
        assert torch.allclose(output[0, 1], embed.W_E[1])
        assert torch.allclose(output[0, 2], embed.W_E[2])

    def test_gradient_flow(self, small_config, sample_tokens):
        """Test that gradients flow through embeddings."""
        embed = EmbedWithoutTorch(small_config)
        output = embed(sample_tokens)
        loss = output.sum()
        loss.backward()
        assert embed.W_E.grad is not None


@pytest.mark.unit
class TestEmbedWithTorch:
    """Tests for EmbedWithTorch."""

    def test_init(self, small_config):
        """Test initialization."""
        embed = EmbedWithTorch(small_config)
        assert embed.embedding.weight.shape == (small_config.d_vocab, small_config.d_model)

    def test_forward(self, small_config, sample_tokens):
        """Test forward pass."""
        embed = EmbedWithTorch(small_config)
        output = embed(sample_tokens)
        assert output.shape == (sample_tokens.shape[0], sample_tokens.shape[1], small_config.d_model)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_batch(self, small_config, sample_batch_tokens):
        """Test forward pass with batch."""
        embed = EmbedWithTorch(small_config)
        output = embed(sample_batch_tokens)
        assert output.shape == (sample_batch_tokens.shape[0], sample_batch_tokens.shape[1], small_config.d_model)

    def test_equivalence_without_torch(self, small_config, sample_tokens):
        """Test that both implementations produce similar outputs."""
        embed1 = EmbedWithoutTorch(small_config)
        embed2 = EmbedWithTorch(small_config)
        
        # Copy weights to make them equivalent
        embed2.embedding.weight.data = embed1.W_E.data.clone()
        
        output1 = embed1(sample_tokens)
        output2 = embed2(sample_tokens)
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_gradient_flow(self, small_config, sample_tokens):
        """Test that gradients flow through embeddings."""
        embed = EmbedWithTorch(small_config)
        output = embed(sample_tokens)
        loss = output.sum()
        loss.backward()
        assert embed.embedding.weight.grad is not None


@pytest.mark.unit
class TestUnembedWithoutTorch:
    """Tests for UnembedWithoutTorch."""

    def test_init(self, small_config):
        """Test initialization."""
        unembed = UnembedWithoutTorch(small_config)
        assert unembed.W_U.shape == (small_config.d_model, small_config.d_vocab)
        assert isinstance(unembed.W_U, torch.nn.Parameter)

    def test_forward(self, small_config):
        """Test forward pass."""
        unembed = UnembedWithoutTorch(small_config)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, small_config.d_model)
        output = unembed(residual)
        assert output.shape == (batch_size, seq_len, small_config.d_vocab)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_output_shape(self, small_config):
        """Test output shape matches expected."""
        unembed = UnembedWithoutTorch(small_config)
        residual = torch.randn(1, 10, small_config.d_model)
        output = unembed(residual)
        assert output.shape == (1, 10, small_config.d_vocab)

    def test_gradient_flow(self, small_config):
        """Test that gradients flow through unembedding."""
        unembed = UnembedWithoutTorch(small_config)
        residual = torch.randn(1, 5, small_config.d_model, requires_grad=True)
        output = unembed(residual)
        loss = output.sum()
        loss.backward()
        assert unembed.W_U.grad is not None
        assert residual.grad is not None


@pytest.mark.unit
class TestUnembedWithTorch:
    """Tests for UnembedWithTorch."""

    def test_init(self, small_config):
        """Test initialization."""
        unembed = UnembedWithTorch(small_config)
        assert unembed.linear.weight.shape == (small_config.d_vocab, small_config.d_model)
        assert unembed.linear.bias is None  # No bias in unembedding

    def test_forward(self, small_config):
        """Test forward pass."""
        unembed = UnembedWithTorch(small_config)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, small_config.d_model)
        output = unembed(residual)
        assert output.shape == (batch_size, seq_len, small_config.d_vocab)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_equivalence_without_torch(self, small_config):
        """Test that both implementations produce similar outputs."""
        unembed1 = UnembedWithoutTorch(small_config)
        unembed2 = UnembedWithTorch(small_config)
        
        # Copy weights to make them equivalent
        unembed2.linear.weight.data = unembed1.W_U.data.clone().t()
        
        residual = torch.randn(1, 5, small_config.d_model)
        output1 = unembed1(residual)
        output2 = unembed2(residual)
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_gradient_flow(self, small_config):
        """Test that gradients flow through unembedding."""
        unembed = UnembedWithTorch(small_config)
        residual = torch.randn(1, 5, small_config.d_model, requires_grad=True)
        output = unembed(residual)
        loss = output.sum()
        loss.backward()
        assert unembed.linear.weight.grad is not None
        assert residual.grad is not None

