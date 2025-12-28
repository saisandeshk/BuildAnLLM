"""Property-based tests for model output properties."""

import pytest
import torch
from pretraining.model.model import TransformerModel


@pytest.mark.property
class TestModelProperties:
    """Property-based tests for model outputs."""

    def test_output_shape_consistency(self, small_config, sample_tokens):
        """Property: Model output shape should match input shape."""
        model = TransformerModel(small_config, use_einops=True)
        result = model(sample_tokens)
        # Handle tuple return (logits, cache) - extract just logits
        logits = result[0] if isinstance(result, tuple) else result
        
        batch_size, seq_len = sample_tokens.shape
        assert logits.shape == (batch_size, seq_len, small_config.d_vocab)

    def test_deterministic_output(self, small_config, sample_tokens):
        """Property: Model should produce deterministic outputs with same seed."""
        torch.manual_seed(42)
        model1 = TransformerModel(small_config, use_einops=True)
        result1 = model1(sample_tokens)
        logits1 = result1[0] if isinstance(result1, tuple) else result1
        
        torch.manual_seed(42)
        model2 = TransformerModel(small_config, use_einops=True)
        result2 = model2(sample_tokens)
        logits2 = result2[0] if isinstance(result2, tuple) else result2
        
        assert torch.allclose(logits1, logits2, atol=1e-5)

    def test_gradient_flow(self, small_config, sample_tokens):
        """Property: Gradients should flow through model."""
        model = TransformerModel(small_config, use_einops=True)
        result = model(sample_tokens)
        # Handle tuple return (logits, cache) - extract just logits
        logits = result[0] if isinstance(result, tuple) else result
        loss = logits.sum()
        loss.backward()
        
        # Check that some parameters have gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_logits_range(self, small_config, sample_tokens):
        """Property: Logits should be in reasonable range."""
        model = TransformerModel(small_config, use_einops=True)
        result = model(sample_tokens)
        # Handle tuple return (logits, cache) - extract just logits
        logits = result[0] if isinstance(result, tuple) else result
        
        # Logits shouldn't be extreme
        assert torch.all(torch.abs(logits) < 100)  # Reasonable range
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

