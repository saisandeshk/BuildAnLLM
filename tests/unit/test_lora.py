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


@pytest.mark.unit
class TestLoraWithCache:
    """Tests for LoRA models with KV cache support."""

    def test_lora_forward_without_cache(self, model_with_einops):
        """Test LoRA model forward pass without cache."""
        model_lora = convert_model_to_lora(
            model_with_einops,
            rank=8,
            alpha=8.0,
            dropout=0.0
        )
        
        batch_size, seq_len = 2, 5
        tokens = torch.randint(0, model_with_einops.cfg.d_vocab, (batch_size, seq_len))
        
        result = model_lora(tokens)
        # Should return (logits, cache) tuple
        assert isinstance(result, tuple)
        assert len(result) == 2
        logits, cache = result
        assert logits.shape == (batch_size, seq_len, model_with_einops.cfg.d_vocab)
        assert isinstance(cache, list)
        assert len(cache) == len(model_lora.blocks)

    def test_lora_forward_with_cache(self, model_with_einops):
        """Test LoRA model forward pass with cache."""
        model_lora = convert_model_to_lora(
            model_with_einops,
            rank=8,
            alpha=8.0,
            dropout=0.0
        )
        
        batch_size = 2
        tokens1 = torch.randint(0, model_with_einops.cfg.d_vocab, (batch_size, 5))
        tokens2 = torch.randint(0, model_with_einops.cfg.d_vocab, (batch_size, 1))
        
        # First forward pass
        result1 = model_lora(tokens1, cache=None, start_pos=0)
        logits1, cache1 = result1
        
        # Second forward pass with cache
        result2 = model_lora(tokens2, cache=cache1, start_pos=5)
        logits2, cache2 = result2
        
        assert logits2.shape == (batch_size, 1, model_with_einops.cfg.d_vocab)
        assert isinstance(cache2, list)
        assert len(cache2) == len(model_lora.blocks)
        # Cache should accumulate
        assert cache2[0][0].shape[1] == cache1[0][0].shape[1] + 1  # One more token

    def test_lora_attention_with_cache(self, model_with_einops):
        """Test LoRA attention module accepts cache parameter."""
        model_lora = convert_model_to_lora(
            model_with_einops,
            rank=8,
            alpha=8.0,
            dropout=0.0
        )
        
        attn = model_lora.blocks[0].attn
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, model_with_einops.cfg.d_model)
        
        # Test without cache
        output1, cache1 = attn(residual, cache=None, start_pos=0)
        assert output1.shape == (batch_size, seq_len, model_with_einops.cfg.d_model)
        assert isinstance(cache1, tuple)
        assert len(cache1) == 2
        
        # Test with cache
        residual2 = torch.randn(batch_size, 1, model_with_einops.cfg.d_model)
        output2, cache2 = attn(residual2, cache=cache1, start_pos=seq_len)
        assert output2.shape == (batch_size, 1, model_with_einops.cfg.d_model)
        assert isinstance(cache2, tuple)
        assert len(cache2) == 2
        # Cache should accumulate
        assert cache2[0].shape[1] == cache1[0].shape[1] + 1

    def test_lora_with_rope_and_cache(self, llama_model_with_einops):
        """Test LoRA with RoPE and cache."""
        model_lora = convert_model_to_lora(
            llama_model_with_einops,
            rank=8,
            alpha=8.0,
            dropout=0.0
        )
        
        batch_size = 2
        tokens1 = torch.randint(0, llama_model_with_einops.cfg.d_vocab, (batch_size, 5))
        tokens2 = torch.randint(0, llama_model_with_einops.cfg.d_vocab, (batch_size, 1))
        
        # First forward pass
        result1 = model_lora(tokens1, cache=None, start_pos=0)
        logits1, cache1 = result1
        
        # Second forward pass with cache
        result2 = model_lora(tokens2, cache=cache1, start_pos=5)
        logits2, cache2 = result2
        
        assert logits2.shape == (batch_size, 1, llama_model_with_einops.cfg.d_vocab)


@pytest.mark.unit
class TestLoraDevicePlacement:
    """Tests for LoRA device placement."""

    def test_create_lora_matrices_with_device(self):
        """Test creating LoRA matrices on specific device."""
        # Test with CPU device
        weight_cpu = torch.randn(128, 256)
        lora_A, lora_B, scaling, dropout_layer = create_lora_matrices(
            weight_cpu.shape, rank=8, alpha=8.0, dropout=0.0, weight=weight_cpu
        )
        assert lora_A.device.type == 'cpu'
        assert lora_B.device.type == 'cpu'
        
        # Test with MPS if available
        if torch.backends.mps.is_available():
            weight_mps = torch.randn(128, 256).to('mps')
            lora_A_mps, lora_B_mps, _, _ = create_lora_matrices(
                weight_mps.shape, rank=8, alpha=8.0, dropout=0.0, weight=weight_mps
            )
            assert lora_A_mps.device.type == 'mps'
            assert lora_B_mps.device.type == 'mps'

    def test_lora_matrices_match_weight_device(self, model_with_einops):
        """Test that LoRA matrices are created on the same device as weights."""
        # Move model to device first
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        model = model_with_einops.to(device)
        model_lora = convert_model_to_lora(model, rank=8, alpha=8.0, dropout=0.0)
        
        # Check that LoRA matrices are on the same device as base weights
        for param_name in ['W_Q', 'W_K', 'W_V', 'W_O']:
            if hasattr(model_lora.blocks[0].attn, param_name):
                base_weight = getattr(model_lora.blocks[0].attn, param_name)
                lora_A = getattr(model_lora.blocks[0].attn, f'{param_name}_lora_A')
                lora_B = getattr(model_lora.blocks[0].attn, f'{param_name}_lora_B')
                
                assert base_weight.device.type == device.type, f"Base weight for {param_name} is on {base_weight.device.type}, expected {device.type}"
                assert lora_A.device.type == device.type, f"LoRA A for {param_name} is on {lora_A.device.type}, expected {device.type}"
                assert lora_B.device.type == device.type, f"LoRA B for {param_name} is on {lora_B.device.type}, expected {device.type}"

    def test_einsum_with_lora_device_handling(self):
        """Test that einsum_with_lora handles device mismatches."""
        # Create tensors on different devices if MPS available
        if torch.backends.mps.is_available():
            x = torch.randn(2, 5, 256).to('mps')
            weight = torch.randn(4, 64, 256).to('mps')
            lora_A = torch.randn(4, 8, 256).to('cpu')  # Different device
            lora_B = torch.randn(4, 64, 8).to('cpu')  # Different device
            scaling = 1.0
            dropout_layer = torch.nn.Identity()
            
            # This should handle device mismatch
            output = einsum_with_lora(x, weight, 
                                     "batch posn d_model, n_heads d_head d_model -> batch posn n_heads d_head",
                                     lora_A, lora_B, scaling, dropout_layer)
            assert output.device.type == 'mps'  # Should match input device


@pytest.mark.unit
class TestModelReturnFormat:
    """Tests for model return format handling."""

    def test_model_returns_tuple(self, model_with_einops):
        """Test that model always returns tuple (logits, cache)."""
        batch_size, seq_len = 2, 5
        tokens = torch.randint(0, model_with_einops.cfg.d_vocab, (batch_size, seq_len))
        
        result = model_with_einops(tokens)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected tuple of length 2, got {len(result)}"
        
        logits, cache = result
        assert isinstance(logits, torch.Tensor)
        assert isinstance(cache, list)
        assert len(cache) == len(model_with_einops.blocks)

    def test_extract_logits_from_tuple(self, model_with_einops):
        """Test helper function to extract logits from tuple."""
        batch_size, seq_len = 2, 5
        tokens = torch.randint(0, model_with_einops.cfg.d_vocab, (batch_size, seq_len))
        
        result = model_with_einops(tokens)
        
        # Test extraction logic used in training code
        logits = result[0] if isinstance(result, tuple) else result
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (batch_size, seq_len, model_with_einops.cfg.d_vocab)

    def test_lora_model_returns_tuple(self, model_with_einops):
        """Test that LoRA model returns tuple (logits, cache)."""
        model_lora = convert_model_to_lora(model_with_einops, rank=8, alpha=8.0, dropout=0.0)
        
        batch_size, seq_len = 2, 5
        tokens = torch.randint(0, model_with_einops.cfg.d_vocab, (batch_size, seq_len))
        
        result = model_lora(tokens)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2
        
        logits, cache = result
        assert isinstance(logits, torch.Tensor)
        assert isinstance(cache, list)

