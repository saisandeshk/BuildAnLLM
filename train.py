# /// script
# dependencies = ["torch", "einops", "jaxtyping", "numpy", "tiktoken", "sentencepiece"]
# ///

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from jaxtyping import Float
from torch import Tensor
from config import GPTConfig
from layernorm import LayerNormWithEinops, LayerNormWithoutEinops, LayerNormWithTorch
from embed import EmbedWithoutTorch, EmbedWithTorch
from positional_embedding import PosEmbedWithEinops, PosEmbedWithoutEinops
from attention import AttentionWithEinops, AttentionWithoutEinops
from mlp import MLPWithEinops, MLPWithoutEinops
from transformer_block import TransformerBlockWithEinops, TransformerBlockWithoutEinops
from gpt import GPTWithEinops, GPTWithoutEinops
from tokenizer import (
    CharacterTokenizer,
    CharacterTokenizerWithTorch,
    BPETokenizer,
    SentencePieceTokenizer,
    TorchTokenizer,
)

device = torch.device(
    "mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu"
)

# Load training data
with open("training.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Initialize config
cfg = GPTConfig()

TOKENIZER_TYPE = "bpe"

# Create tokenizer based on type
if TOKENIZER_TYPE == "character":
    tokenizer = CharacterTokenizer(text)
elif TOKENIZER_TYPE == "character_torch":
    tokenizer = CharacterTokenizerWithTorch(text)
elif TOKENIZER_TYPE == "bpe":
    tokenizer = BPETokenizer(text)
elif TOKENIZER_TYPE == "sentencepiece":
    tokenizer = SentencePieceTokenizer(
        text, vocab_size=min(cfg.d_vocab, 10000))
else:
    raise ValueError(f"Unknown tokenizer type: {TOKENIZER_TYPE}")

print(f"Using {TOKENIZER_TYPE} tokenizer")
print(f"Vocabulary size: {tokenizer.vocab_size}")

sample_text = text[:100]
encoded = tokenizer.encode(sample_text)
decoded = tokenizer.decode(encoded)
print(f"\nOriginal (first 50 chars): {repr(sample_text[:50])}")
print(f"Decoded (first 50 chars):   {repr(decoded[:50])}")
print(f"Perfect match: {sample_text == decoded}")
