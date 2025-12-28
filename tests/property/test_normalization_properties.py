"""Property-based tests for normalization mathematical properties."""

import pytest
import torch
from pretraining.normalization.layernorm import LayerNorm
from pretraining.normalization.rmsnorm import RMSNorm


@pytest.mark.property
@pytest.mark.parametrize("use_einops", [True, False])
class TestNormalizationProperties:
    """Property-based tests for normalization layers."""

    def test_layernorm_output_statistics(self, small_config, use_einops):
        """Property: LayerNorm output should have mean ~0, std ~1."""
        ln = LayerNorm(small_config, use_einops=use_einops)
        # Reset to identity for testing
        ln.w.data.fill_(1.0)
        ln.b.data.fill_(0.0)
        
        residual = torch.randn(1, 1, small_config.d_model)
        output = ln(residual)
        
        # After normalization, mean should be ~0, std ~1
        output_mean = output.mean().item()
        output_std = output.std().item()
        assert abs(output_mean) < 0.1
        assert abs(output_std - 1.0) < 0.1

    def test_layernorm_scale_invariance(self, small_config, use_einops):
        """Property: LayerNorm is scale-invariant (up to learnable scale)."""
        ln = LayerNorm(small_config, use_einops=use_einops)
        ln.w.data.fill_(1.0)
        ln.b.data.fill_(0.0)
        
        residual = torch.randn(1, 5, small_config.d_model)
        output1 = ln(residual)
        output2 = ln(residual * 2)
        
        # After normalization, outputs should be similar (scale doesn't matter)
        # But not identical due to numerical precision
        assert torch.allclose(output1, output2, atol=0.1)

    def test_rmsnorm_output_statistics(self, llama_config, use_einops):
        """Property: RMSNorm output should have RMS ~1."""
        rms = RMSNorm(llama_config, use_einops=use_einops)
        rms.w.data.fill_(1.0)
        
        residual = torch.randn(1, 1, llama_config.d_model)
        output = rms(residual)
        
        # Compute RMS of output
        output_rms = torch.sqrt((output ** 2).mean()).item()
        # Should be close to 1 (after scaling by w=1)
        assert abs(output_rms - 1.0) < 0.1

    def test_rmsnorm_scale_invariance(self, llama_config, use_einops):
        """Property: RMSNorm is scale-invariant."""
        rms = RMSNorm(llama_config, use_einops=use_einops)
        rms.w.data.fill_(1.0)
        
        residual = torch.randn(1, 5, llama_config.d_model)
        output1 = rms(residual)
        output2 = rms(residual * 2)
        
        # Should be similar after normalization
        assert torch.allclose(output1, output2, atol=0.1)

