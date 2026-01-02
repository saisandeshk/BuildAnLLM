"""Integration tests for model checkpoint loading."""

import pytest
import torch
import os
from config import PositionalEncoding
from pretraining.model.model_loader import load_model_from_checkpoint
from pretraining.model.model import TransformerModel


@pytest.mark.integration
class TestModelLoading:
    """Tests for model checkpoint loading."""

    def test_save_and_load(self, model_with_einops, temp_checkpoint_dir, device):
        """Test saving and loading model."""
        checkpoint_path = os.path.join(temp_checkpoint_dir, "test_model.pt")
        
        # Save model
        checkpoint = {
            "model_state_dict": model_with_einops.state_dict(),
            "cfg": model_with_einops.cfg.to_dict(),
            "model_type": "with_einops",
            "tokenizer_type": "character",
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Load model
        loaded_model, loaded_cfg, loaded_checkpoint = load_model_from_checkpoint(
            checkpoint_path, device
        )
        
        assert isinstance(loaded_model, TransformerModel)
        assert loaded_cfg.d_model == model_with_einops.cfg.d_model
        assert loaded_checkpoint["model_type"] == "with_einops"

    def test_load_without_einops(self, model_without_einops, temp_checkpoint_dir, device):
        """Test loading model without einops."""
        checkpoint_path = os.path.join(temp_checkpoint_dir, "test_model_no_einops.pt")
        
        checkpoint = {
            "model_state_dict": model_without_einops.state_dict(),
            "cfg": model_without_einops.cfg.to_dict(),
            "model_type": "without_einops",
            "tokenizer_type": "character",
        }
        torch.save(checkpoint, checkpoint_path)
        
        loaded_model, _, _ = load_model_from_checkpoint(checkpoint_path, device)
        assert isinstance(loaded_model, TransformerModel)
        assert not loaded_model.use_einops

    def test_load_different_architecture(self, llama_model_with_einops, temp_checkpoint_dir, device):
        """Test loading different architecture."""
        checkpoint_path = os.path.join(temp_checkpoint_dir, "test_llama.pt")
        
        checkpoint = {
            "model_state_dict": llama_model_with_einops.state_dict(),
            "cfg": llama_model_with_einops.cfg.to_dict(),
            "model_type": "with_einops",
            "tokenizer_type": "character",
        }
        torch.save(checkpoint, checkpoint_path)
        
        loaded_model, loaded_cfg, _ = load_model_from_checkpoint(checkpoint_path, device)
        assert loaded_cfg.positional_encoding == PositionalEncoding.ROPE

    def test_load_missing_checkpoint(self, device):
        """Test loading non-existent checkpoint."""
        with pytest.raises(FileNotFoundError):
            load_model_from_checkpoint("nonexistent.pt", device)

    def test_load_with_shape_mismatch(self, model_with_einops, temp_checkpoint_dir, device):
        """Test loading with shape mismatch."""
        checkpoint_path = os.path.join(temp_checkpoint_dir, "test_mismatch.pt")
        
        # Create checkpoint with wrong shape
        wrong_state_dict = model_with_einops.state_dict().copy()
        wrong_state_dict['embed.W_E'] = torch.randn(100, 128)  # Wrong shape
        
        checkpoint = {
            "model_state_dict": wrong_state_dict,
            "cfg": model_with_einops.cfg.to_dict(),
            "model_type": "with_einops",
            "tokenizer_type": "character",
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Should still load but skip mismatched keys
        loaded_model, _, _ = load_model_from_checkpoint(checkpoint_path, device)
        assert isinstance(loaded_model, TransformerModel)
