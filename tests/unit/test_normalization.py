"""Unit tests for normalization layers."""

import pytest
import torch
from pretraining.normalization.layernorm import LayerNorm, create_norm_layer
from pretraining.normalization.rmsnorm import RMSNorm
from config import Normalization


@pytest.mark.unit
@pytest.mark.parametrize("use_einops", [True, False])
class TestLayerNorm:
    """Tests for LayerNorm."""

    def test_init(self, small_config, use_einops):
        """Test initialization."""
        ln = LayerNorm(small_config, use_einops=use_einops)
        assert ln.w.shape == (small_config.d_model,)
        assert ln.b.shape == (small_config.d_model,)
        assert torch.allclose(ln.w, torch.ones(small_config.d_model))
        assert torch.allclose(ln.b, torch.zeros(small_config.d_model))

    def test_forward(self, small_config, use_einops):
        """Test forward pass."""
        ln = LayerNorm(small_config, use_einops=use_einops)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, small_config.d_model)
        output = ln(residual)
        assert output.shape == residual.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_compute_mean(self, small_config, use_einops):
        """Test mean computation."""
        ln = LayerNorm(small_config, use_einops=use_einops)
        residual = torch.randn(2, 5, small_config.d_model)
        mean = ln._compute_mean(residual)
        assert mean.shape == (2, 5, 1)
        # Check that mean is computed correctly
        expected_mean = residual.mean(dim=-1, keepdim=True)
        assert torch.allclose(mean, expected_mean, atol=1e-5)

    def test_compute_variance(self, small_config, use_einops):
        """Test variance computation."""
        ln = LayerNorm(small_config, use_einops=use_einops)
        residual = torch.randn(2, 5, small_config.d_model)
        variance = ln._compute_variance(residual)
        assert variance.shape == (2, 5, 1)
        # Check that variance is computed correctly (unbiased=False)
        expected_variance = residual.var(dim=-1, keepdim=True, unbiased=False)
        assert torch.allclose(variance, expected_variance, atol=1e-5)

    def test_normalization_properties(self, small_config, use_einops):
        """Test that normalization centers and scales correctly."""
        ln = LayerNorm(small_config, use_einops=use_einops)
        # Reset to identity (w=1, b=0) for testing
        ln.w.data.fill_(1.0)
        ln.b.data.fill_(0.0)
        
        residual = torch.randn(1, 1, small_config.d_model)
        output = ln(residual)
        
        # After normalization (with identity scale/shift), mean should be ~0, std ~1
        output_mean = output.mean()
        output_std = output.std()
        assert abs(output_mean.item()) < 0.1  # Should be close to 0
        assert abs(output_std.item() - 1.0) < 0.1  # Should be close to 1

    def test_gradient_flow(self, small_config, use_einops):
        """Test that gradients flow through normalization."""
        ln = LayerNorm(small_config, use_einops=use_einops)
        residual = torch.randn(1, 5, small_config.d_model, requires_grad=True)
        output = ln(residual)
        loss = output.sum()
        loss.backward()
        assert ln.w.grad is not None
        assert ln.b.grad is not None
        assert residual.grad is not None

    def test_learnable_parameters(self, small_config, use_einops):
        """Test that scale and shift parameters are learnable."""
        ln = LayerNorm(small_config, use_einops=use_einops)
        residual = torch.randn(1, 5, small_config.d_model)
        output1 = ln(residual)
        
        # Modify parameters
        ln.w.data.fill_(2.0)
        ln.b.data.fill_(1.0)
        output2 = ln(residual)
        
        # Outputs should be different
        assert not torch.allclose(output1, output2)

@pytest.mark.unit
class TestLayerNormEquivalence:
    """Tests for LayerNorm equivalence between implementations."""

    def test_equivalence_einops_vs_torch(self, small_config):
        """Test that einops and non-einops implementations are equivalent."""
        ln1 = LayerNorm(small_config, use_einops=True)
        ln2 = LayerNorm(small_config, use_einops=False)
        
        # Copy parameters
        ln2.w.data = ln1.w.data.clone()
        ln2.b.data = ln1.b.data.clone()
        
        residual = torch.randn(2, 5, small_config.d_model)
        output1 = ln1(residual)
        output2 = ln2(residual)
        
        assert torch.allclose(output1, output2, atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("use_einops", [True, False])
class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_init(self, small_config, use_einops):
        """Test initialization."""
        rms = RMSNorm(small_config, use_einops=use_einops)
        assert rms.w.shape == (small_config.d_model,)
        assert torch.allclose(rms.w, torch.ones(small_config.d_model))
        # RMSNorm has no bias
        assert not hasattr(rms, 'b')

    def test_forward(self, small_config, use_einops):
        """Test forward pass."""
        rms = RMSNorm(small_config, use_einops=use_einops)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, small_config.d_model)
        output = rms(residual)
        assert output.shape == residual.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_compute_rms(self, small_config, use_einops):
        """Test RMS computation."""
        rms = RMSNorm(small_config, use_einops=use_einops)
        residual = torch.randn(2, 5, small_config.d_model)
        rms_value = rms._compute_rms(residual)
        assert rms_value.shape == (2, 5, 1)
        # RMS = sqrt(mean(x^2) + eps)
        expected_rms = torch.sqrt((residual ** 2).mean(dim=-1, keepdim=True) + small_config.layer_norm_eps)
        assert torch.allclose(rms_value, expected_rms, atol=1e-5)

    def test_no_bias(self, small_config, use_einops):
        """Test that RMSNorm has no bias term."""
        rms = RMSNorm(small_config, use_einops=use_einops)
        assert not hasattr(rms, 'b')

    def test_gradient_flow(self, small_config, use_einops):
        """Test that gradients flow through RMSNorm."""
        rms = RMSNorm(small_config, use_einops=use_einops)
        residual = torch.randn(1, 5, small_config.d_model, requires_grad=True)
        output = rms(residual)
        loss = output.sum()
        loss.backward()
        assert rms.w.grad is not None
        assert residual.grad is not None

@pytest.mark.unit
class TestRMSNormEquivalence:
    """Tests for RMSNorm equivalence between implementations."""

    def test_equivalence_einops_vs_torch(self, small_config):
        """Test that einops and non-einops implementations are equivalent."""
        rms1 = RMSNorm(small_config, use_einops=True)
        rms2 = RMSNorm(small_config, use_einops=False)
        
        # Copy parameters
        rms2.w.data = rms1.w.data.clone()
        
        residual = torch.randn(2, 5, small_config.d_model)
        output1 = rms1(residual)
        output2 = rms2(residual)
        
        assert torch.allclose(output1, output2, atol=1e-5)


@pytest.mark.unit
class TestCreateNormLayer:
    """Tests for create_norm_layer factory function."""

    def test_create_layernorm(self, small_config):
        """Test creating LayerNorm."""
        ln = create_norm_layer(small_config, use_einops=True)
        assert isinstance(ln, LayerNorm)

    def test_create_rmsnorm(self, llama_config):
        """Test creating RMSNorm."""
        rms = create_norm_layer(llama_config, use_einops=True)
        assert isinstance(rms, RMSNorm)

    def test_einops_flag(self, small_config):
        """Test einops flag is respected."""
        ln1 = create_norm_layer(small_config, use_einops=True)
        ln2 = create_norm_layer(small_config, use_einops=False)
        assert ln1.use_einops == True
        assert ln2.use_einops == False

