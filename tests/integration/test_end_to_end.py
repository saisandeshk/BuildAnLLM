"""End-to-end integration tests."""

import pytest
import torch
import os
from config import ModelConfig
from pretraining.data.dataset import TransformerDataset
from pretraining.model.model import TransformerModel
from pretraining.training.trainer import TransformerTrainer
from pretraining.model.model_loader import load_model_from_checkpoint
from inference.sampler import TransformerSampler
from pretraining.training.training_args import TransformerTrainingArgs


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Tests for complete training/inference workflows."""

    def test_train_save_load_infer(self, small_config, sample_text, character_tokenizer, device, temp_checkpoint_dir):
        """Test complete workflow: train → save → load → infer."""
        # Step 1: Create dataset
        dataset = TransformerDataset(sample_text, small_config, tokenizer_type="character")
        X_train, Y_train = dataset.get_train_data()
        X_val, Y_val = dataset.get_val_data()
        
        # Step 2: Create and train model
        model = TransformerModel(small_config, use_einops=True)
        model = model.to(device)
        
        training_args = TransformerTrainingArgs(
            batch_size=4,
            epochs=1,
            max_steps_per_epoch=5,
            lr=1e-3,
            eval_iters=2,
            save_dir=temp_checkpoint_dir
        )
        
        trainer = TransformerTrainer(
            model=model,
            args=training_args,
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            device=device
        )
        
        # Train for a few steps
        trainer.train()
        
        # Step 3: Save model
        checkpoint_path = os.path.join(temp_checkpoint_dir, "final_model.pt")
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "cfg": model.cfg.to_dict(),
            "model_type": "with_einops",
            "tokenizer_type": "character",
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Step 4: Load model
        loaded_model, loaded_cfg, _ = load_model_from_checkpoint(checkpoint_path, device)
        assert isinstance(loaded_model, TransformerModel)
        
        # Step 5: Inference
        sampler = TransformerSampler(loaded_model, character_tokenizer, device)
        prompt = "Hello"
        generated = sampler.sample(prompt, max_new_tokens=5)
        assert isinstance(generated, str)
        assert len(generated) > len(prompt)

    def test_different_architectures(self, sample_text, character_tokenizer, device, temp_checkpoint_dir):
        """Test workflow with different architectures."""
        for arch_config in [ModelConfig.gpt_small(), ModelConfig.llama_small(), ModelConfig.olmo_small()]:
            # Create dataset
            dataset = TransformerDataset(sample_text, arch_config, tokenizer_type="character")
            X_train, Y_train = dataset.get_train_data()
            X_val, Y_val = dataset.get_val_data()
            
            # Create model
            model = TransformerModel(arch_config, use_einops=True)
            model = model.to(device)
            
            # Quick training
            training_args = TransformerTrainingArgs(
                batch_size=2,
                epochs=1,
                max_steps_per_epoch=2,
                lr=1e-3,
                eval_iters=1,
                save_dir=temp_checkpoint_dir
            )
            
            trainer = TransformerTrainer(
                model=model,
                args=training_args,
                X_train=X_train,
                Y_train=Y_train,
                X_val=X_val,
                Y_val=Y_val,
                device=device
            )
            
            # Train briefly
            trainer.train()
            
            # Test inference
            sampler = TransformerSampler(model, character_tokenizer, device)
            generated = sampler.sample("Test", max_new_tokens=3)
            assert isinstance(generated, str)

