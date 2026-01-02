from dataclasses import dataclass
from enum import Enum
from typing import Union


class PositionalEncoding(str, Enum):
    LEARNED = "learned"  # Learned positional embeddings (GPT style)
    ROPE = "rope"  # Rotary Position Embedding (LLaMA style)
    ALIBI = "alibi"  # Attention with Linear Biases (OLMo style)
    NONE = "none"  # No positional encoding


class Normalization(str, Enum):
    LAYERNORM = "layernorm"  # LayerNorm (GPT, OLMo style)
    RMSNORM = "rmsnorm"  # RMSNorm (LLaMA style)


class Activation(str, Enum):
    GELU = "gelu"  # GELU activation (GPT style)
    SWIGLU = "swiglu"  # SwiGLU activation (LLaMA, OLMo style)


class RouterType(str, Enum):
    TOP_K = "top_k"  # Standard top-k routing
    TOP_K_WITH_SHARED = "top_k_with_shared"  # Top-k with shared experts (DeepSeek-style)


@dataclass
class ModelConfig:
    # Model dimensions
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_kv_heads: int = None  # Number of KV heads for GQA/MQA (None = n_heads for MHA)
    n_layers: int = 12

    # Configurable model components (if None, default to GPT-style settings)
    positional_encoding: Union[PositionalEncoding, None] = None
    normalization: Union[Normalization, None] = None
    activation: Union[Activation, None] = None

    # LLaMA-specific
    rope_theta: float = 10000.0  # Base frequency for RoPE

    # MoE (Mixture of Experts) configuration
    use_moe: bool = False  # Enable/disable MoE
    num_experts: int = 8  # Number of expert MLPs
    num_experts_per_tok: int = 2  # Top-k experts to activate per token
    use_shared_experts: bool = False  # Enable shared experts (DeepSeek-style)
    num_shared_experts: int = 2  # Number of always-active shared experts
    router_type: Union[RouterType, None] = None  # Routing strategy
    load_balancing_loss_weight: float = 0.01  # Weight for load balancing loss
    expert_capacity_factor: float = 1.25  # Capacity factor for expert load balancing

    def __post_init__(self):
        """Set defaults if not explicitly provided."""
        if self.positional_encoding is None:
            self.positional_encoding = PositionalEncoding.LEARNED

        if self.normalization is None:
            self.normalization = Normalization.LAYERNORM

        if self.activation is None:
            self.activation = Activation.GELU

        if self.positional_encoding == PositionalEncoding.ROPE and self.d_head % 2 != 0:
            raise ValueError("d_head must be even when using RoPE.")

        # Set default router_type based on use_shared_experts if not explicitly set
        if self.use_moe and self.router_type is None:
            if self.use_shared_experts:
                self.router_type = RouterType.TOP_K_WITH_SHARED
            else:
                self.router_type = RouterType.TOP_K

        # Set default n_kv_heads for GQA/MQA support
        # If None, default to n_heads (MHA behavior - backward compatible)
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        
        # Validate n_kv_heads for GQA/MQA
        if self.n_kv_heads > self.n_heads:
            raise ValueError(f"n_kv_heads ({self.n_kv_heads}) cannot be greater than n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads}) for GQA")

    @classmethod
    def gpt_small(cls):
        """Small GPT config for faster training/testing (good for Mac)"""
        return cls(
            d_model=256,
            n_heads=4,
            n_layers=4,
            n_ctx=256,
            d_head=64,
            d_mlp=1024,
            d_vocab=50257,  # Will be updated by tokenizer
        )

    @classmethod
    def gpt_medium(cls):
        """Medium GPT config (between small and full)"""
        return cls(
            d_model=512,
            n_heads=8,
            n_layers=6,
            n_ctx=512,
            d_head=64,
            d_mlp=2048,
            d_vocab=50257,
        )

    @classmethod
    def gpt_full(cls):
        """Full GPT-2 size config"""
        return cls(
            d_model=768,
            n_heads=12,
            n_layers=12,
            n_ctx=1024,
            d_head=64,
            d_mlp=3072,
            d_vocab=50257,
        )

    @classmethod
    def llama_small(cls):
        """Small LLaMA config for faster training/testing"""
        return cls(
            positional_encoding=PositionalEncoding.ROPE,
            normalization=Normalization.RMSNORM,
            activation=Activation.SWIGLU,
            d_model=256,
            n_heads=4,
            n_layers=4,
            n_ctx=256,
            d_head=64,
            d_mlp=1024,
            d_vocab=50257,  # Will be updated by tokenizer
            rope_theta=10000.0,
        )

    @classmethod
    def llama_medium(cls):
        """Medium LLaMA config (between small and full)"""
        return cls(
            positional_encoding=PositionalEncoding.ROPE,
            normalization=Normalization.RMSNORM,
            activation=Activation.SWIGLU,
            d_model=512,
            n_heads=8,
            n_layers=6,
            n_ctx=512,
            d_head=64,
            d_mlp=2048,
            d_vocab=50257,
            rope_theta=10000.0,
        )

    @classmethod
    def llama_full(cls):
        """Full LLaMA config"""
        return cls(
            positional_encoding=PositionalEncoding.ROPE,
            normalization=Normalization.RMSNORM,
            activation=Activation.SWIGLU,
            d_model=768,
            n_heads=12,
            n_layers=12,
            n_ctx=1024,
            d_head=64,
            d_mlp=3072,
            d_vocab=50257,
            rope_theta=10000.0,
        )

    @classmethod
    def olmo_small(cls):
        """Small OLMo config for faster training/testing"""
        return cls(
            positional_encoding=PositionalEncoding.ALIBI,
            normalization=Normalization.LAYERNORM,
            activation=Activation.SWIGLU,
            d_model=256,
            n_heads=4,
            n_layers=4,
            n_ctx=256,
            d_head=64,
            d_mlp=1024,
            d_vocab=50257,  # Will be updated by tokenizer
        )

    @classmethod
    def olmo_medium(cls):
        """Medium OLMo config (between small and full)"""
        return cls(
            positional_encoding=PositionalEncoding.ALIBI,
            normalization=Normalization.LAYERNORM,
            activation=Activation.SWIGLU,
            d_model=512,
            n_heads=8,
            n_layers=6,
            n_ctx=512,
            d_head=64,
            d_mlp=2048,
            d_vocab=50257,
        )

    @classmethod
    def olmo_full(cls):
        """Full OLMo config"""
        return cls(
            positional_encoding=PositionalEncoding.ALIBI,
            normalization=Normalization.LAYERNORM,
            activation=Activation.SWIGLU,
            d_model=768,
            n_heads=12,
            n_layers=12,
            n_ctx=1024,
            d_head=64,
            d_mlp=3072,
            d_vocab=50257,
        )

    def to_dict(self):
        """Convert config to dict with enum values as strings for serialization."""
        from dataclasses import asdict
        config_dict = asdict(self)
        # Handle positional_encoding
        pos_enc = config_dict.get("positional_encoding")
        if isinstance(pos_enc, Enum):
            config_dict["positional_encoding"] = pos_enc.value
        # Handle normalization
        norm = config_dict.get("normalization")
        if isinstance(norm, Enum):
            config_dict["normalization"] = norm.value
        # Handle activation
        act = config_dict.get("activation")
        if isinstance(act, Enum):
            config_dict["activation"] = act.value
        # Handle router_type
        router = config_dict.get("router_type")
        if isinstance(router, Enum):
            config_dict["router_type"] = router.value
        # None values are already None, no need to set them again
        return config_dict

    @classmethod
    def from_dict(cls, config_dict):
        """Reconstruct config from dict with proper enum reconstruction."""
        config_dict = dict(config_dict)
        from dataclasses import fields
        allowed = {field.name for field in fields(cls)}
        config_dict = {key: value for key, value in config_dict.items() if key in allowed}
        # Convert string enum values back to enum instances
        if "positional_encoding" in config_dict:
            pos_enc = config_dict["positional_encoding"]
            if isinstance(pos_enc, str):
                config_dict["positional_encoding"] = PositionalEncoding(
                    pos_enc)
            elif config_dict["positional_encoding"] is None:
                pass  # Keep None
        if "normalization" in config_dict:
            if isinstance(config_dict["normalization"], str):
                config_dict["normalization"] = Normalization(
                    config_dict["normalization"])
            elif config_dict["normalization"] is None:
                pass  # Keep None
        if "activation" in config_dict:
            if isinstance(config_dict["activation"], str):
                config_dict["activation"] = Activation(
                    config_dict["activation"])
            elif config_dict["activation"] is None:
                pass  # Keep None
        if "router_type" in config_dict:
            router = config_dict["router_type"]
            if isinstance(router, str):
                config_dict["router_type"] = RouterType(router)
            elif router is None:
                pass  # Keep None
        # Ensure backward compatibility: if use_moe is False and router_type not set, keep router_type as None
        if not config_dict.get("use_moe", False) and "router_type" not in config_dict:
            config_dict["router_type"] = None
        return cls(**config_dict)

    @classmethod
    def from_ui_dict(cls, model_config: dict):
        """Create ModelConfig from UI config dict (helper for Streamlit UI)."""
        return cls(
            d_model=model_config["d_model"],
            n_heads=model_config["n_heads"],
            # Default to MHA if not specified
            n_kv_heads=model_config.get("n_kv_heads", model_config["n_heads"]),
            n_layers=model_config["n_layers"],
            n_ctx=model_config["n_ctx"],
            d_head=model_config["d_head"],
            d_mlp=model_config["d_mlp"],
            positional_encoding=PositionalEncoding(
                model_config["positional_encoding"]),
            normalization=Normalization(model_config["normalization"]),
            activation=Activation(model_config["activation"]),
            rope_theta=model_config.get("rope_theta", 10000.0),
            use_moe=model_config.get("use_moe", False),
            num_experts=model_config.get("num_experts", 8),
            num_experts_per_tok=model_config.get("num_experts_per_tok", 2),
            use_shared_experts=model_config.get("use_shared_experts", False),
            num_shared_experts=model_config.get("num_shared_experts", 2),
            router_type=RouterType(model_config.get(
                "router_type", "top_k")) if model_config.get("use_moe", False) else None,
            load_balancing_loss_weight=model_config.get(
                "load_balancing_loss_weight", 0.01),
            expert_capacity_factor=model_config.get(
                "expert_capacity_factor", 1.25),
        )
