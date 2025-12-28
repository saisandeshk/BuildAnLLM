"""Shared fixtures for pytest tests."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

from config import ModelConfig, Architecture, PositionalEncoding, Normalization, Activation, RouterType
from pretraining.tokenization.tokenizer import (
    CharacterTokenizer,
    CharacterTokenizerWithTorch,
    SimpleBPETokenizer,
    BPETokenizer,
    SentencePieceTokenizer,
)
from pretraining.model.model import TransformerModel
from pretraining.training.training_args import TransformerTrainingArgs
from finetuning.training.finetuning_args import FinetuningArgs


@pytest.fixture(scope="session")
def device():
    """Test device - use CPU for deterministic tests."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def set_seed():
    """Set random seed for deterministic tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def small_config():
    """Small model config for fast tests."""
    return ModelConfig.gpt_small()


@pytest.fixture
def gpt_config():
    """GPT architecture config."""
    return ModelConfig.gpt_small()


@pytest.fixture
def llama_config():
    """LLaMA architecture config."""
    return ModelConfig.llama_small()


@pytest.fixture
def olmo_config():
    """OLMo architecture config."""
    return ModelConfig.olmo_small()


@pytest.fixture
def moe_config():
    """MoE config for testing."""
    cfg = ModelConfig.llama_small()
    cfg.use_moe = True
    cfg.num_experts = 4
    cfg.num_experts_per_tok = 2
    cfg.use_shared_experts = False
    cfg.router_type = RouterType.TOP_K
    return cfg


@pytest.fixture
def moe_with_shared_config():
    """MoE config with shared experts."""
    cfg = ModelConfig.llama_small()
    cfg.use_moe = True
    cfg.num_experts = 4
    cfg.num_experts_per_tok = 2
    cfg.use_shared_experts = True
    cfg.num_shared_experts = 1
    cfg.router_type = RouterType.TOP_K_WITH_SHARED
    return cfg


@pytest.fixture
def gqa_config():
    """GQA config (Grouped Query Attention)."""
    cfg = ModelConfig.llama_small()
    cfg.n_heads = 8
    cfg.n_kv_heads = 2  # 4:1 ratio
    return cfg


@pytest.fixture
def mqa_config():
    """MQA config (Multi-Query Attention)."""
    cfg = ModelConfig.llama_small()
    cfg.n_heads = 8
    cfg.n_kv_heads = 1  # All Q heads share single K/V head
    return cfg


@pytest.fixture
def sample_text():
    """Sample text for tokenizer training."""
    # Include uppercase letters, lowercase letters, numbers, and common punctuation
    return "Hello World! This is a test. The quick brown fox jumps over the lazy dog. ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789. What is Python? Explain transformers. " * 10


@pytest.fixture
def sample_prompt_response_pairs():
    """Sample prompt/response pairs for SFT testing."""
    return [
        ("What is Python?", "Python is a programming language."),
        ("Explain transformers.", "Transformers are neural network architectures."),
        ("What is attention?", "Attention allows models to focus on relevant parts of input."),
    ]


@pytest.fixture
def character_tokenizer(sample_text):
    """Character tokenizer fixture."""
    return CharacterTokenizer(sample_text)


@pytest.fixture
def character_tokenizer_torch(sample_text):
    """Character tokenizer with torch fixture."""
    return CharacterTokenizerWithTorch(sample_text)


@pytest.fixture
def simple_bpe_tokenizer(sample_text):
    """Simple BPE tokenizer fixture."""
    return SimpleBPETokenizer(sample_text, vocab_size=100)


@pytest.fixture
def bpe_tokenizer():
    """BPE tokenizer (tiktoken) fixture."""
    return BPETokenizer()


@pytest.fixture
def sentencepiece_tokenizer(sample_text):
    """SentencePiece tokenizer fixture."""
    return SentencePieceTokenizer(sample_text, vocab_size=100)


@pytest.fixture
def model_with_einops(small_config, device):
    """Model with einops implementation."""
    model = TransformerModel(small_config, use_einops=True)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def model_without_einops(small_config, device):
    """Model without einops implementation."""
    model = TransformerModel(small_config, use_einops=False)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def llama_model_with_einops(llama_config, device):
    """LLaMA model with einops."""
    model = TransformerModel(llama_config, use_einops=True)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def olmo_model_with_einops(olmo_config, device):
    """OLMo model with einops."""
    model = TransformerModel(olmo_config, use_einops=True)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def moe_model(moe_config, device):
    """MoE model fixture."""
    model = TransformerModel(moe_config, use_einops=True)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def training_args():
    """Training arguments fixture."""
    return TransformerTrainingArgs(
        batch_size=4,
        epochs=1,
        max_steps_per_epoch=10,
        lr=1e-3,
        weight_decay=1e-2,
        eval_iters=5,
    )


@pytest.fixture
def finetuning_args():
    """Finetuning arguments fixture."""
    return FinetuningArgs(
        batch_size=2,
        epochs=1,
        max_steps_per_epoch=5,
        lr=1e-5,
        weight_decay=0.01,
        eval_iters=3,
        use_lora=False,
    )


@pytest.fixture
def temp_checkpoint_dir():
    """Temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_tokens(character_tokenizer):
    """Sample token IDs for testing."""
    text = "Hello world"
    tokens = character_tokenizer.encode(text)
    return torch.tensor([tokens], dtype=torch.long)


@pytest.fixture
def sample_batch_tokens(character_tokenizer):
    """Sample batch of token IDs."""
    texts = ["Hello world", "Test text"]
    batch_tokens = [character_tokenizer.encode(text) for text in texts]
    max_len = max(len(t) for t in batch_tokens)
    # Pad to same length
    padded = [t + [0] * (max_len - len(t)) for t in batch_tokens]
    return torch.tensor(padded, dtype=torch.long)


@pytest.fixture
def sample_csv_file(sample_prompt_response_pairs, temp_checkpoint_dir):
    """Create a temporary CSV file for SFT dataset testing."""
    import pandas as pd
    
    csv_path = os.path.join(temp_checkpoint_dir, "test_sft_data.csv")
    df = pd.DataFrame(sample_prompt_response_pairs, columns=["prompt", "response"])
    df.to_csv(csv_path, index=False)
    return csv_path

