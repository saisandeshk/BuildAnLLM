"""Unit tests for utility functions."""

import pytest
import torch
from utils import get_device, print_state_dict_warnings
from pretraining.utils import (
    extract_model_output_and_aux_loss,
    extract_mlp_output_and_aux_loss,
    add_aux_loss_to_main_loss,
)


@pytest.mark.unit
class TestGetDevice:
    """Tests for get_device function."""

    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']


@pytest.mark.unit
class TestPrintStateDictWarnings:
    """Tests for print_state_dict_warnings function."""

    def test_no_warnings(self, capsys):
        """Test with no warnings."""
        print_state_dict_warnings([], [])
        captured = capsys.readouterr()
        assert "Warning" not in captured.out

    def test_unexpected_keys(self, capsys):
        """Test with unexpected keys."""
        unexpected = ['layer1.weight', 'layer2.bias']
        print_state_dict_warnings(unexpected, [])
        captured = capsys.readouterr()
        assert "unexpected" in captured.out.lower()
        assert "layer1.weight" in captured.out

    def test_missing_keys(self, capsys):
        """Test with missing keys."""
        missing = ['layer1.weight', 'layer2.bias']
        print_state_dict_warnings([], missing)
        captured = capsys.readouterr()
        assert "missing" in captured.out.lower()
        assert "layer1.weight" in captured.out

    def test_many_keys(self, capsys):
        """Test with many keys (truncation)."""
        unexpected = [f'key{i}' for i in range(10)]
        print_state_dict_warnings(unexpected, [])
        captured = capsys.readouterr()
        assert "... and" in captured.out or "more" in captured.out


@pytest.mark.unit
class TestExtractModelOutputAndAuxLoss:
    """Tests for extract_model_output_and_aux_loss function."""

    def test_logits_only(self):
        """Test with logits only."""
        logits = torch.randn(2, 5, 100)
        logits_result, aux_loss = extract_model_output_and_aux_loss(logits)
        assert torch.allclose(logits_result, logits)
        assert aux_loss is None

    def test_logits_with_aux_loss(self):
        """Test with logits and aux_loss."""
        logits = torch.randn(2, 5, 100)
        aux_loss = torch.tensor(1.0)
        result = (logits, aux_loss)
        logits_result, aux_loss_result = extract_model_output_and_aux_loss(result)
        assert torch.allclose(logits_result, logits)
        assert aux_loss_result == aux_loss

    def test_logits_with_cache(self):
        """Test with logits and cache."""
        logits = torch.randn(2, 5, 100)
        cache = [(torch.randn(2, 5, 4, 64), torch.randn(2, 5, 4, 64))]
        result = (logits, cache)
        logits_result, aux_loss = extract_model_output_and_aux_loss(result)
        assert torch.allclose(logits_result, logits)
        assert aux_loss is None

    def test_logits_with_cache_and_aux_loss(self):
        """Test with logits, cache, and aux_loss."""
        logits = torch.randn(2, 5, 100)
        cache = [(torch.randn(2, 5, 4, 64), torch.randn(2, 5, 4, 64))]
        aux_loss = torch.tensor(1.0)
        result = (logits, cache, aux_loss)
        logits_result, aux_loss_result = extract_model_output_and_aux_loss(result)
        assert torch.allclose(logits_result, logits)
        assert aux_loss_result == aux_loss


@pytest.mark.unit
class TestExtractMLPOutputAndAuxLoss:
    """Tests for extract_mlp_output_and_aux_loss function."""

    def test_output_only(self):
        """Test with output only."""
        output = torch.randn(2, 5, 256)
        output_result, aux_loss = extract_mlp_output_and_aux_loss(output)
        assert torch.allclose(output_result, output)
        assert aux_loss is None

    def test_output_with_aux_loss(self):
        """Test with output and aux_loss."""
        output = torch.randn(2, 5, 256)
        aux_loss = torch.tensor(1.0)
        result = (output, aux_loss)
        output_result, aux_loss_result = extract_mlp_output_and_aux_loss(result)
        assert torch.allclose(output_result, output)
        assert aux_loss_result == aux_loss


@pytest.mark.unit
class TestAddAuxLossToMainLoss:
    """Tests for add_aux_loss_to_main_loss function."""

    def test_no_aux_loss(self, model_with_einops):
        """Test with no aux_loss."""
        loss = torch.tensor(2.0)
        result = add_aux_loss_to_main_loss(loss, None, model_with_einops)
        assert result == loss

    def test_with_aux_loss(self, moe_model):
        """Test with aux_loss."""
        loss = torch.tensor(2.0)
        aux_loss = torch.tensor(0.5)
        result = add_aux_loss_to_main_loss(loss, aux_loss, moe_model)
        expected = loss + aux_loss * moe_model.cfg.load_balancing_loss_weight
        assert torch.allclose(result, expected)

    def test_without_model_config(self):
        """Test without model config."""
        loss = torch.tensor(2.0)
        aux_loss = torch.tensor(0.5)
        # Create a simple object without cfg
        class SimpleModel:
            pass
        model = SimpleModel()
        result = add_aux_loss_to_main_loss(loss, aux_loss, model)
        # Should return loss unchanged if no config
        assert result == loss

