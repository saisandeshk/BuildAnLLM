"""Unit tests for MLP classes."""

import pytest
import torch
from pretraining.mlp.mlp import MLP, MLPSwiGLU, create_mlp_layer, MoEMLP
from config import Activation


@pytest.mark.unit
@pytest.mark.parametrize("use_einops", [True, False])
class TestMLP:
    """Tests for MLP (GELU activation)."""

    def test_init(self, small_config, use_einops):
        """Test initialization."""
        mlp = MLP(small_config, use_einops=use_einops)
        assert mlp.W_in.shape == (small_config.d_model, small_config.d_mlp)
        assert mlp.W_out.shape == (small_config.d_mlp, small_config.d_model)
        assert mlp.b_in.shape == (small_config.d_mlp,)
        assert mlp.b_out.shape == (small_config.d_model,)

    def test_forward(self, small_config, use_einops):
        """Test forward pass."""
        mlp = MLP(small_config, use_einops=use_einops)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, small_config.d_model)
        output = mlp(residual)
        assert output.shape == residual.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_compute_hidden(self, small_config, use_einops):
        """Test hidden layer computation."""
        mlp = MLP(small_config, use_einops=use_einops)
        residual = torch.randn(2, 5, small_config.d_model)
        hidden = mlp._compute_hidden(residual)
        assert hidden.shape == (2, 5, small_config.d_mlp)

    def test_project_output(self, small_config, use_einops):
        """Test output projection."""
        mlp = MLP(small_config, use_einops=use_einops)
        hidden = torch.randn(2, 5, small_config.d_mlp)
        output = mlp._project_output(hidden)
        assert output.shape == (2, 5, small_config.d_model)

    def test_gradient_flow(self, small_config, use_einops):
        """Test that gradients flow through MLP."""
        mlp = MLP(small_config, use_einops=use_einops)
        residual = torch.randn(1, 5, small_config.d_model, requires_grad=True)
        output = mlp(residual)
        loss = output.sum()
        loss.backward()
        assert mlp.W_in.grad is not None
        assert mlp.W_out.grad is not None
        assert residual.grad is not None

@pytest.mark.unit
class TestMLPEquivalence:
    """Tests for MLP equivalence between implementations."""

    def test_equivalence_einops_vs_torch(self, small_config):
        """Test that einops and non-einops implementations are equivalent."""
        mlp1 = MLP(small_config, use_einops=True)
        mlp2 = MLP(small_config, use_einops=False)
        
        # Copy parameters
        mlp2.W_in.data = mlp1.W_in.data.clone()
        mlp2.W_out.data = mlp1.W_out.data.clone()
        mlp2.b_in.data = mlp1.b_in.data.clone()
        mlp2.b_out.data = mlp1.b_out.data.clone()
        
        residual = torch.randn(2, 5, small_config.d_model)
        output1 = mlp1(residual)
        output2 = mlp2(residual)
        assert torch.allclose(output1, output2, atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("use_einops", [True, False])
class TestMLPSwiGLU:
    """Tests for MLPSwiGLU (SwiGLU activation)."""

    def test_init(self, llama_config, use_einops):
        """Test initialization."""
        mlp = MLPSwiGLU(llama_config, use_einops=use_einops)
        assert mlp.W_gate.shape == (llama_config.d_model, llama_config.d_mlp)
        assert mlp.W_up.shape == (llama_config.d_model, llama_config.d_mlp)
        assert mlp.W_out.shape == (llama_config.d_mlp, llama_config.d_model)
        assert mlp.b_gate.shape == (llama_config.d_mlp,)
        assert mlp.b_up.shape == (llama_config.d_mlp,)
        assert mlp.b_out.shape == (llama_config.d_model,)

    def test_forward(self, llama_config, use_einops):
        """Test forward pass."""
        mlp = MLPSwiGLU(llama_config, use_einops=use_einops)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, llama_config.d_model)
        output = mlp(residual)
        assert output.shape == residual.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_compute_gate(self, llama_config, use_einops):
        """Test gate computation."""
        mlp = MLPSwiGLU(llama_config, use_einops=use_einops)
        residual = torch.randn(2, 5, llama_config.d_model)
        gate = mlp._compute_gate(residual)
        assert gate.shape == (2, 5, llama_config.d_mlp)

    def test_compute_up(self, llama_config, use_einops):
        """Test up branch computation."""
        mlp = MLPSwiGLU(llama_config, use_einops=use_einops)
        residual = torch.randn(2, 5, llama_config.d_model)
        up = mlp._compute_up(residual)
        assert up.shape == (2, 5, llama_config.d_mlp)

    def test_gradient_flow(self, llama_config, use_einops):
        """Test that gradients flow through SwiGLU MLP."""
        mlp = MLPSwiGLU(llama_config, use_einops=use_einops)
        residual = torch.randn(1, 5, llama_config.d_model, requires_grad=True)
        output = mlp(residual)
        loss = output.sum()
        loss.backward()
        assert mlp.W_gate.grad is not None
        assert mlp.W_up.grad is not None
        assert mlp.W_out.grad is not None
        assert residual.grad is not None


@pytest.mark.unit
class TestMoEMLP:
    """Tests for MoE MLP."""

    def test_init(self, moe_config):
        """Test MoE initialization."""
        mlp = MoEMLP(moe_config, use_einops=True)
        assert mlp.num_experts == moe_config.num_experts
        assert mlp.num_experts_per_tok == moe_config.num_experts_per_tok
        assert len(mlp.experts) == moe_config.num_experts
        assert mlp.router.weight.shape == (moe_config.num_experts, moe_config.d_model)

    def test_forward(self, moe_config):
        """Test MoE forward pass."""
        mlp = MoEMLP(moe_config, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, moe_config.d_model)
        mlp.train()  # Enable training to get aux_loss
        output, aux_loss = mlp(residual)
        assert output.shape == residual.shape
        assert aux_loss is not None
        assert aux_loss.item() >= 0

    def test_forward_eval(self, moe_config):
        """Test MoE forward pass in eval mode."""
        mlp = MoEMLP(moe_config, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, moe_config.d_model)
        mlp.eval()
        output, aux_loss = mlp(residual)
        assert output.shape == residual.shape
        assert aux_loss is None  # No aux loss in eval mode

    def test_routing(self, moe_config):
        """Test routing computation."""
        mlp = MoEMLP(moe_config, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, moe_config.d_model)
        router_probs, top_k_probs, top_k_indices = mlp._compute_routing(residual)
        assert router_probs.shape == (batch_size, seq_len, moe_config.num_experts)
        assert top_k_probs.shape == (batch_size, seq_len, moe_config.num_experts_per_tok)
        assert top_k_indices.shape == (batch_size, seq_len, moe_config.num_experts_per_tok)
        # Check that probabilities sum to 1
        assert torch.allclose(router_probs.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)

    def test_load_balancing_loss(self, moe_config):
        """Test load balancing loss computation."""
        mlp = MoEMLP(moe_config, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, moe_config.d_model)
        router_probs, top_k_probs, top_k_indices = mlp._compute_routing(residual)
        aux_loss = mlp._compute_load_balancing_loss(router_probs, top_k_indices, batch_size, seq_len)
        assert aux_loss.item() >= 0
        assert aux_loss.requires_grad

    def test_shared_experts(self, moe_with_shared_config):
        """Test MoE with shared experts."""
        mlp = MoEMLP(moe_with_shared_config, use_einops=True)
        assert mlp.use_shared_experts
        assert mlp.shared_experts is not None
        assert len(mlp.shared_experts) == moe_with_shared_config.num_shared_experts
        
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, moe_with_shared_config.d_model)
        output, aux_loss = mlp(residual)
        assert output.shape == residual.shape


@pytest.mark.unit
class TestCreateMLPLayer:
    """Tests for create_mlp_layer factory function."""

    def test_create_mlp_gelu(self, small_config):
        """Test creating GELU MLP."""
        mlp = create_mlp_layer(small_config, use_einops=True)
        assert isinstance(mlp, MLP)

    def test_create_mlp_swiglu(self, llama_config):
        """Test creating SwiGLU MLP."""
        mlp = create_mlp_layer(llama_config, use_einops=True)
        assert isinstance(mlp, MLPSwiGLU)

    def test_create_moe_mlp(self, moe_config):
        """Test creating MoE MLP."""
        mlp = create_mlp_layer(moe_config, use_einops=True)
        assert isinstance(mlp, MoEMLP)

