import torch
import torch.nn as nn
from jaxtyping import Float, Int, Tensor
from embed import EmbedWithoutTorch, EmbedWithTorch
from positional_embedding import PosEmbedWithEinops, PosEmbedWithoutEinops
from transformer_block import TransformerBlockWithEinops, TransformerBlockWithoutEinops
from layernorm import LayerNormWithEinops, LayerNormWithoutEinops


class GPTWithEinops(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = EmbedWithoutTorch(cfg)
        self.pos_embed = PosEmbedWithEinops(cfg)
        self.blocks = nn.ModuleList([
            TransformerBlockWithEinops(cfg) for _ in range(cfg.n_layers)
        ])
        self.ln_f = LayerNormWithEinops(cfg)
        self.unembed = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.unembed, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        # Token embeddings
        residual = self.embed(tokens)

        # Positional embeddings
        residual = residual + self.pos_embed(tokens)

        # Transformer blocks
        for block in self.blocks:
            residual = block(residual)

        # Final layer norm
        residual = self.ln_f(residual)

        # Unembedding to logits
        # residual: [batch, position, d_model]
        # unembed: [d_model, d_vocab]
        # logits: [batch, position, d_vocab]
        logits = torch.matmul(residual, self.unembed)

        return logits


class GPTWithoutEinops(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = EmbedWithTorch(cfg)
        self.pos_embed = PosEmbedWithoutEinops(cfg)
        self.blocks = nn.ModuleList([
            TransformerBlockWithoutEinops(cfg) for _ in range(cfg.n_layers)
        ])
        self.ln_f = LayerNormWithoutEinops(cfg)
        self.unembed = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.unembed, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        # Token embeddings
        residual = self.embed(tokens)

        # Positional embeddings
        residual = residual + self.pos_embed(tokens)

        # Transformer blocks
        for block in self.blocks:
            residual = block(residual)

        # Final layer norm
        residual = self.ln_f(residual)

        # Unembedding to logits
        # residual: [batch, position, d_model]
        # unembed: [d_model, d_vocab]
        # logits: [batch, position, d_vocab]
        logits = torch.matmul(residual, self.unembed)

        return logits
