"""Unit tests for TransformerModel class."""

import pytest
import torch
from pretraining.model.model import TransformerModel, _aggregate_aux_losses


@pytest.mark.unit
@pytest.mark.parametrize("use_einops", [True, False])
class TestTransformerModel:
    """Tests for TransformerModel."""

    def test_init(self, small_config, use_einops):
        """Test initialization."""
        model = TransformerModel(small_config, use_einops=use_einops)
        assert hasattr(model, 'embed')
        assert hasattr(model, 'blocks')
        assert len(model.blocks) == small_config.n_layers
        assert hasattr(model, 'ln_f')
        assert hasattr(model, 'unembed')

    def test_forward(self, small_config, sample_tokens, use_einops):
        """Test forward pass."""
        model = TransformerModel(small_config, use_einops=use_einops)
        result = model(sample_tokens)
        # Handle tuple return (logits, cache) - extract just logits
        logits = result[0] if isinstance(result, tuple) else result
        batch_size, seq_len = sample_tokens.shape
        assert logits.shape == (batch_size, seq_len, small_config.d_vocab)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_forward_with_cache(self, small_config, use_einops):
        """Test forward pass with KV cache."""
        model = TransformerModel(small_config, use_einops=use_einops)
        batch_size = 2
        
        # First forward pass
        tokens1 = torch.randint(0, small_config.d_vocab, (batch_size, 5))
        logits1, cache1 = model(tokens1, cache=None, start_pos=0)
        
        # Second forward pass with cache
        tokens2 = torch.randint(0, small_config.d_vocab, (batch_size, 1))
        logits2, cache2 = model(tokens2, cache=cache1, start_pos=5)
        
        assert logits2.shape == (batch_size, 1, small_config.d_vocab)

    def test_cache_matches_full_logits(self, small_config, use_einops):
        """Cached logits should match full-sequence logits for the last token."""
        model = TransformerModel(small_config, use_einops=use_einops)
        model.eval()
        batch_size, seq_len = 2, 6
        tokens = torch.randint(0, small_config.d_vocab, (batch_size, seq_len))

        with torch.no_grad():
            result_full = model(tokens, cache=None, start_pos=0)
            logits_full = result_full[0] if isinstance(result_full, tuple) else result_full

            tokens_prefix = tokens[:, :-1]
            tokens_last = tokens[:, -1:]
            result_prefix = model(tokens_prefix, cache=None, start_pos=0)
            logits_prefix, cache = result_prefix
            result_cached = model(tokens_last, cache=cache, start_pos=seq_len - 1)
            logits_cached = result_cached[0] if isinstance(result_cached, tuple) else result_cached

        torch.testing.assert_close(
            logits_cached, logits_full[:, -1:, :], rtol=1e-4, atol=1e-5
        )

    def test_gradient_flow(self, small_config, sample_tokens, use_einops):
        """Test that gradients flow through model."""
        model = TransformerModel(small_config, use_einops=use_einops)
        result = model(sample_tokens)
        # Handle tuple return (logits, cache) - extract just logits
        logits = result[0] if isinstance(result, tuple) else result
        loss = logits.sum()
        loss.backward()
        # Check that some parameters have gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad


@pytest.mark.unit
class TestTransformerModelArchitectures:
    """Tests for different architectures."""

    def test_gpt_model(self, gpt_config, sample_tokens):
        """Test GPT model."""
        model = TransformerModel(gpt_config, use_einops=True)
        assert model.pos_embed is not None  # GPT has learned positional embeddings
        result = model(sample_tokens)
        # Handle tuple return (logits, cache) - extract just logits
        logits = result[0] if isinstance(result, tuple) else result
        assert logits.shape == (sample_tokens.shape[0], sample_tokens.shape[1], gpt_config.d_vocab)

    def test_llama_model(self, llama_config, sample_tokens):
        """Test LLaMA model."""
        model = TransformerModel(llama_config, use_einops=True)
        assert model.pos_embed is None  # LLaMA uses RoPE, not learned embeddings
        assert model.rope is not None
        result = model(sample_tokens)
        # Handle tuple return (logits, cache) - extract just logits
        logits = result[0] if isinstance(result, tuple) else result
        assert logits.shape == (sample_tokens.shape[0], sample_tokens.shape[1], llama_config.d_vocab)

    def test_olmo_model(self, olmo_config, sample_tokens):
        """Test OLMo model."""
        model = TransformerModel(olmo_config, use_einops=True)
        assert model.pos_embed is None  # OLMo uses ALiBi, not learned embeddings
        assert model.alibi is not None
        result = model(sample_tokens)
        # Handle tuple return (logits, cache) - extract just logits
        logits = result[0] if isinstance(result, tuple) else result
        assert logits.shape == (sample_tokens.shape[0], sample_tokens.shape[1], olmo_config.d_vocab)


@pytest.mark.unit
class TestTransformerModelWithMoE:
    """Tests for TransformerModel with MoE."""

    def test_forward_with_moe(self, moe_config):
        """Test forward pass with MoE."""
        model = TransformerModel(moe_config, use_einops=True)
        batch_size, seq_len = 2, 5
        tokens = torch.randint(0, moe_config.d_vocab, (batch_size, seq_len))
        model.train()  # Enable training to get aux_loss
        result = model(tokens)
        
        # MoE models return (logits, cache, aux_loss) when cache is returned
        assert isinstance(result, tuple)
        if len(result) == 3:
            logits, cache, aux_loss = result
        elif len(result) == 2:
            # Could be (logits, cache) or (logits, aux_loss) - check types
            if isinstance(result[1], (list, tuple)) and len(result[1]) > 0:
                logits, cache = result
                aux_loss = None
            else:
                logits, aux_loss = result
        else:
            logits = result
            aux_loss = None
        
        assert logits.shape == (batch_size, seq_len, moe_config.d_vocab)
        assert aux_loss is not None, "MoE model should return aux_loss in training mode"
        assert aux_loss.item() >= 0

    def test_forward_with_moe_eval(self, moe_config):
        """Test forward pass with MoE in eval mode."""
        model = TransformerModel(moe_config, use_einops=True)
        batch_size, seq_len = 2, 5
        tokens = torch.randint(0, moe_config.d_vocab, (batch_size, seq_len))
        model.eval()
        result = model(tokens)
        
        # In eval mode, aux_loss may be None
        if isinstance(result, tuple):
            logits, aux_loss = result
            assert logits.shape == (batch_size, seq_len, moe_config.d_vocab)
        else:
            logits = result
            assert logits.shape == (batch_size, seq_len, moe_config.d_vocab)


@pytest.mark.unit
class TestAggregateAuxLosses:
    """Tests for _aggregate_aux_losses function."""

    def test_aggregate_losses(self):
        """Test aggregating multiple aux losses."""
        losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
        result = _aggregate_aux_losses(losses)
        assert result is not None
        assert result.item() == 6.0

    def test_empty_list(self):
        """Test with empty list."""
        result = _aggregate_aux_losses([])
        assert result is None

    def test_none_losses(self):
        """Test with None values."""
        losses = [torch.tensor(1.0), None, torch.tensor(2.0)]
        result = _aggregate_aux_losses(losses)
        # Should handle None gracefully
        assert result is not None
