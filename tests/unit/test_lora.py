"""Unit tests for LoRA utilities and wrappers."""

import pytest
import torch
from finetuning.peft.lora_utils import (
    convert_model_to_lora,
    get_lora_parameters,
    count_lora_parameters,
)
from finetuning.peft.lora_wrappers import (
    create_lora_matrices,
    apply_lora_to_attention,
    apply_lora_to_mlp,
    einsum_with_lora,
)


@pytest.mark.unit
class TestCreateLoraMatrices:
    """Tests for create_lora_matrices function."""

    def test_create_2d_matrices(self):
        """Test creating LoRA matrices for 2D weights."""
        weight_shape = (128, 256)
        rank = 8
        alpha = 8.0
        dropout = 0.0
        
        lora_A, lora_B, scaling, dropout_layer = create_lora_matrices(
            weight_shape, rank, alpha, dropout
        )
        
        assert lora_A.shape == (rank, 256)  # [rank, in_dim]
        assert lora_B.shape == (128, rank)  # [out_dim, rank]
        assert scaling == alpha / rank
        assert isinstance(dropout_layer, torch.nn.Identity)

    def test_create_3d_matrices(self):
        """Test creating LoRA matrices for 3D weights (attention)."""
        weight_shape = (4, 64, 256)  # [n_heads, d_head, d_model]
        rank = 8
        alpha = 8.0
        dropout = 0.1
        
        lora_A, lora_B, scaling, dropout_layer = create_lora_matrices(
            weight_shape, rank, alpha, dropout
        )
        
        assert lora_A.shape == (4, rank, 256)  # [n_heads, rank, d_model]
        assert lora_B.shape == (4, 64, rank)  # [n_heads, d_head, rank]
        assert scaling == alpha / rank
        assert isinstance(dropout_layer, torch.nn.Dropout)


@pytest.mark.unit
class TestApplyLoraToAttention:
    """Tests for apply_lora_to_attention function."""

    def test_apply_lora(self, llama_model_with_einops):
        """Test applying LoRA to attention layer."""
        attn = llama_model_with_einops.blocks[0].attn
        apply_lora_to_attention(attn, rank=8, alpha=8.0, dropout=0.0)
        
        # Check that LoRA matrices were created
        assert hasattr(attn, 'W_Q_lora_A')
        assert hasattr(attn, 'W_Q_lora_B')
        assert hasattr(attn, 'W_K_lora_A')
        assert hasattr(attn, 'W_K_lora_B')
        assert hasattr(attn, 'W_V_lora_A')
        assert hasattr(attn, 'W_V_lora_B')
        assert hasattr(attn, 'W_O_lora_A')
        assert hasattr(attn, 'W_O_lora_B')
        
        # Check that original weights are frozen
        assert not attn.W_Q.requires_grad
        assert not attn.W_K.requires_grad
        assert not attn.W_V.requires_grad
        assert not attn.W_O.requires_grad
        
        # Check that LoRA weights are trainable
        assert attn.W_Q_lora_A.requires_grad
        assert attn.W_Q_lora_B.requires_grad


@pytest.mark.unit
class TestApplyLoraToMLP:
    """Tests for apply_lora_to_mlp function."""

    def test_apply_lora_gelu(self, model_with_einops):
        """Test applying LoRA to GELU MLP."""
        mlp = model_with_einops.blocks[0].mlp
        apply_lora_to_mlp(mlp, rank=8, alpha=8.0, dropout=0.0)
        
        # Check that LoRA matrices were created
        assert hasattr(mlp, 'W_in_lora_A')
        assert hasattr(mlp, 'W_in_lora_B')
        assert hasattr(mlp, 'W_out_lora_A')
        assert hasattr(mlp, 'W_out_lora_B')
        
        # Check that original weights are frozen
        assert not mlp.W_in.requires_grad
        assert not mlp.W_out.requires_grad

    def test_apply_lora_swiglu(self, llama_model_with_einops):
        """Test applying LoRA to SwiGLU MLP."""
        mlp = llama_model_with_einops.blocks[0].mlp
        apply_lora_to_mlp(mlp, rank=8, alpha=8.0, dropout=0.0)
        
        # Check that LoRA matrices were created for SwiGLU
        assert hasattr(mlp, 'W_gate_lora_A')
        assert hasattr(mlp, 'W_gate_lora_B')
        assert hasattr(mlp, 'W_up_lora_A')
        assert hasattr(mlp, 'W_up_lora_B')
        assert hasattr(mlp, 'W_out_lora_A')
        assert hasattr(mlp, 'W_out_lora_B')


@pytest.mark.unit
class TestEinsumWithLora:
    """Tests for einsum_with_lora function."""

    def test_einsum_2d(self):
        """Test einsum with LoRA for 2D weights."""
        x = torch.randn(2, 5, 256)
        weight = torch.randn(128, 256)
        pattern = "batch posn d_model, out_dim d_model -> batch posn out_dim"
        
        rank = 8
        alpha = 8.0
        lora_A = torch.randn(rank, 256)
        lora_B = torch.randn(128, rank)
        scaling = alpha / rank
        dropout_layer = torch.nn.Identity()
        
        output = einsum_with_lora(x, weight, pattern, lora_A, lora_B, scaling, dropout_layer)
        assert output.shape == (2, 5, 128)


@pytest.mark.unit
class TestConvertModelToLora:
    """Tests for convert_model_to_lora function."""

    def test_convert_model(self, model_with_einops):
        """Test converting model to LoRA."""
        # Count original trainable parameters
        original_trainable = sum(p.numel() for p in model_with_einops.parameters() if p.requires_grad)
        
        # Convert to LoRA
        model_lora = convert_model_to_lora(
            model_with_einops,
            rank=8,
            alpha=8.0,
            dropout=0.0,
            target_modules="all"
        )
        
        # Count LoRA parameters
        lora_params = get_lora_parameters(model_lora)
        lora_count = sum(p.numel() for p in lora_params)
        
        # LoRA should have fewer trainable parameters
        assert lora_count < original_trainable
        
        # Check that base parameters are frozen
        # Use named_parameters to check parameter names
        base_params_trainable = sum(
            p.numel() for name, p in model_lora.named_parameters()
            if p.requires_grad and 'lora' not in name
        )
        assert base_params_trainable == 0  # All base params should be frozen

    def test_convert_model_attention_only(self, model_with_einops):
        """Test converting only attention layers."""
        model_lora = convert_model_to_lora(
            model_with_einops,
            rank=8,
            alpha=8.0,
            dropout=0.0,
            target_modules="attention"
        )
        
        # Check that attention has LoRA
        assert hasattr(model_lora.blocks[0].attn, 'W_Q_lora_A')
        
        # Check that MLP doesn't have LoRA (if it's not SwiGLU)
        if hasattr(model_lora.blocks[0].mlp, 'W_in'):
            assert not hasattr(model_lora.blocks[0].mlp, 'W_in_lora_A')


@pytest.mark.unit
class TestGetLoraParameters:
    """Tests for get_lora_parameters function."""

    def test_get_lora_params(self, model_with_einops):
        """Test getting LoRA parameters."""
        model_lora = convert_model_to_lora(
            model_with_einops,
            rank=8,
            alpha=8.0,
            dropout=0.0
        )
        
        lora_params = get_lora_parameters(model_lora)
        assert len(lora_params) > 0
        # Check that LoRA parameters are trainable
        assert all(p.requires_grad for p in lora_params)
        # Verify they're actually LoRA parameters by checking names
        # Use id() to compare parameter identity instead of tensor comparison
        lora_param_ids = {id(p) for p in lora_params}
        lora_names = [name for name, p in model_lora.named_parameters() 
                     if id(p) in lora_param_ids]
        assert all('lora_A' in name or 'lora_B' in name for name in lora_names)


@pytest.mark.unit
class TestCountLoraParameters:
    """Tests for count_lora_parameters function."""

    def test_count_params(self, model_with_einops):
        """Test counting LoRA parameters."""
        model_lora = convert_model_to_lora(
            model_with_einops,
            rank=8,
            alpha=8.0,
            dropout=0.0
        )
        
        counts = count_lora_parameters(model_lora)
        assert 'lora' in counts
        assert 'total' in counts
        assert 'trainable' in counts
        assert 'frozen' in counts
        
        assert counts['lora'] > 0
        assert counts['trainable'] == counts['lora']  # Only LoRA params are trainable
        assert counts['frozen'] == counts['total'] - counts['trainable']

