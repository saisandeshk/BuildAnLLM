"""Reusable Streamlit UI components."""

import streamlit as st
from typing import Dict


# Model size presets
MODEL_SIZE_PRESETS = {
    "small": {
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 4,
        "n_ctx": 256,
        "d_head": 64,
        "d_mlp": 1024,
    },
    "medium": {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "n_ctx": 512,
        "d_head": 64,
        "d_mlp": 2048,
    },
    "full": {
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 12,
        "n_ctx": 1024,
        "d_head": 64,
        "d_mlp": 3072,
    },
}


def apply_model_size_preset(size: str, config: Dict) -> None:
    """Apply model size preset to config."""
    preset = MODEL_SIZE_PRESETS[size]
    for key, value in preset.items():
        config[key] = value


def apply_architecture_preset(preset_name: str, config: Dict) -> None:
    """Apply architecture preset (GPT, LLaMA, OLMo) to config."""
    if preset_name == "GPT":
        config["positional_encoding"] = "learned"
        config["normalization"] = "layernorm"
        config["activation"] = "gelu"
        config["tokenizer_type"] = "bpe"
    elif preset_name == "LLAMA":
        config["positional_encoding"] = "rope"
        config["normalization"] = "rmsnorm"
        config["activation"] = "swiglu"
        config["tokenizer_type"] = "sentencepiece"
        config["rope_theta"] = 10000.0
    elif preset_name == "OLMO":
        config["positional_encoding"] = "alibi"
        config["normalization"] = "layernorm"
        config["activation"] = "swiglu"
        config["tokenizer_type"] = "sentencepiece"


def render_model_config_ui() -> Dict:
    """Render model configuration UI and return config dict."""
    # Initialize config if needed
    if "model_config" not in st.session_state:
        st.session_state.model_config = _get_default_config()

    config = st.session_state.model_config

    # Preset buttons
    _render_preset_buttons(config)
    
    # Model components
    _render_model_components(config)
    
    # Model dimensions
    _render_model_dimensions(config)
    
    # Model size selector
    _render_model_size_selector(config)
    
    # RoPE settings (conditional)
    if config["positional_encoding"] == "rope":
        _render_rope_settings(config)

    return config


def _get_default_config() -> Dict:
    """Get default model configuration."""
    return {
        "positional_encoding": "learned",
        "normalization": "layernorm",
        "activation": "gelu",
        "model_size": "small",
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 4,
        "n_ctx": 256,
        "d_head": 64,
        "d_mlp": 1024,
        "rope_theta": 10000.0,
        "tokenizer_type": "bpe",
    }


def _render_preset_buttons(config: Dict) -> None:
    """Render architecture preset buttons."""
    st.subheader("Quick Presets")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        if st.button("🚀 GPT-2", use_container_width=True):
            apply_architecture_preset("GPT", config)
            apply_model_size_preset(config.get("model_size", "small"), config)
            st.rerun()

    with col2:
        if st.button("🦙 LLaMA", use_container_width=True):
            apply_architecture_preset("LLAMA", config)
            apply_model_size_preset(config.get("model_size", "small"), config)
            st.rerun()

    with col3:
        if st.button("🔬 OLMo", use_container_width=True):
            apply_architecture_preset("OLMO", config)
            apply_model_size_preset(config.get("model_size", "small"), config)
            st.rerun()

    with col4:
        with st.expander("ℹ️ About Presets", expanded=False):
            st.markdown(_get_preset_info())


def _render_model_components(config: Dict) -> None:
    """Render model component selectors."""
    st.subheader("Model Components")
    col1, col2, col3 = st.columns(3)

    with col1:
        config["positional_encoding"] = st.selectbox(
            "Positional Encoding",
            ["learned", "rope", "alibi", "none"],
            index=["learned", "rope", "alibi", "none"].index(
                config["positional_encoding"]
            ),
            help="Learned: GPT-style embeddings\nRoPE: Rotary Position Embedding (LLaMA)\nALiBi: Attention with Linear Biases (OLMo)\nNone: No positional encoding"
        )

    with col2:
        config["normalization"] = st.selectbox(
            "Normalization",
            ["layernorm", "rmsnorm"],
            index=["layernorm", "rmsnorm"].index(config["normalization"]),
            help="LayerNorm: GPT/OLMo style\nRMSNorm: LLaMA style (simpler, faster)"
        )

    with col3:
        config["activation"] = st.selectbox(
            "Activation Function",
            ["gelu", "swiglu"],
            index=["gelu", "swiglu"].index(config["activation"]),
            help="GELU: GPT style\nSwiGLU: LLaMA/OLMo style (gated)"
        )


def _render_model_dimensions(config: Dict) -> None:
    """Render model dimension inputs."""
    st.subheader("Model Dimensions")
    col1, col2, col3 = st.columns(3)

    with col1:
        config["d_model"] = st.number_input(
            "d_model (Model Dimension)",
            min_value=64, max_value=4096, value=config["d_model"], step=64,
            help="Hidden dimension size"
        )
        config["n_heads"] = st.number_input(
            "n_heads (Number of Heads)",
            min_value=1, max_value=64, value=config["n_heads"],
            help="Number of attention heads"
        )

    with col2:
        config["n_layers"] = st.number_input(
            "n_layers (Number of Layers)",
            min_value=1, max_value=128, value=config["n_layers"],
            help="Number of transformer layers"
        )
        config["n_ctx"] = st.number_input(
            "n_ctx (Context Length)",
            min_value=64, max_value=8192, value=config["n_ctx"], step=64,
            help="Maximum sequence length"
        )

    with col3:
        config["d_head"] = st.number_input(
            "d_head (Head Dimension)",
            min_value=32, max_value=256, value=config["d_head"], step=32,
            help="Dimension per attention head"
        )
        config["d_mlp"] = st.number_input(
            "d_mlp (MLP Dimension)",
            min_value=128, max_value=16384, value=config["d_mlp"], step=128,
            help="MLP hidden dimension (typically 4x d_model)"
        )


def _render_model_size_selector(config: Dict) -> None:
    """Render model size selector."""
    st.markdown("**Model Size Preset**")
    model_size = st.selectbox(
        "Size",
        ["small", "medium", "full"],
        index=["small", "medium", "full"].index(config.get("model_size", "small")),
        help="Selecting a size automatically updates all model dimensions below."
    )

    if config.get("model_size") != model_size:
        config["model_size"] = model_size
        apply_model_size_preset(model_size, config)
        st.rerun()

    config["model_size"] = model_size


def _render_rope_settings(config: Dict) -> None:
    """Render RoPE-specific settings."""
    config["rope_theta"] = st.number_input(
        "RoPE Theta (Base Frequency)",
        min_value=1000.0, max_value=1000000.0,
        value=config["rope_theta"], step=1000.0, format="%.0f",
        help="Base frequency for RoPE. LLaMA 1/2: 10000, LLaMA 3: 500000"
    )


def _get_preset_info() -> str:
    """Get preset information markdown."""
    return """
    **Preset Configurations:**
    - **GPT-2**: Learned positional embeddings, LayerNorm, GELU activation, BPE tokenizer
    - **LLaMA**: RoPE positional encoding, RMSNorm, SwiGLU activation, SentencePiece tokenizer
    - **OLMo**: ALiBi positional encoding, LayerNorm, SwiGLU activation, SentencePiece tokenizer

    **Model Size:**
    - Controls model dimensions (d_model, n_heads, n_layers, etc.)
    - All presets use the same dimensions for each size
    - Clicking a preset uses the currently selected model size

    **Customization:**
    - All options below can be manually adjusted after selecting a preset
    - Tokenizer is automatically set but can be changed
    """


def generate_model_architecture_diagram(config: Dict) -> str:
    """Generate ASCII art diagram of transformer architecture."""
    n_layers = config.get("n_layers", 4)
    d_model = config.get("d_model", 256)
    n_heads = config.get("n_heads", 4)
    d_mlp = config.get("d_mlp", 1024)
    pos_enc = config.get("positional_encoding", "learned")
    norm = config.get("normalization", "layernorm")
    activation = config.get("activation", "gelu")

    # Map technical names to display names
    pos_enc_display = {
        "learned": "Learned Pos Emb",
        "rope": "RoPE",
        "alibi": "ALiBi",
        "none": "None"
    }.get(pos_enc, pos_enc)

    norm_display = {
        "layernorm": "LayerNorm",
        "rmsnorm": "RMSNorm"
    }.get(norm, norm)

    activation_display = {
        "gelu": "GELU",
        "swiglu": "SwiGLU"
    }.get(activation, activation)

    # Build the diagram
    diagram = []

    # Title
    diagram.append(f"Transformer Architecture ({n_layers} layers, d_model={d_model})")
    diagram.append("="*60)
    diagram.append("")

    # Input section
    diagram.append("                         INPUT")
    diagram.append("                           |")
    diagram.append("                           v")
    diagram.append(f"                   [Token Embeddings]")
    diagram.append(f"                    (vocab → {d_model})")
    diagram.append("                           |")
    if pos_enc == "learned":
        diagram.append("                           +")
        diagram.append(f"                   [{pos_enc_display}]")
        diagram.append("                           |")
    diagram.append("                           v")
    diagram.append("          ┌────────────────────────────────┐")
    diagram.append("          │                                │")
    diagram.append("          │      RESIDUAL STREAM           │")
    diagram.append(f"          │         (d={d_model:4})               │")
    diagram.append("          │                                │")

    # Layers
    for layer_idx in range(n_layers):
        diagram.append(f"          │  ─ ─ ─ Layer {layer_idx + 1} ─ ─ ─ ─ ─ ─     │")
        diagram.append("          │                                │")

        # Attention block
        diagram.append(f"          ├──────> [{norm_display}] ─────┐      │")
        diagram.append("          │                         │      │")
        diagram.append("          │       [Multi-Head       │      │")
        diagram.append("          │        Attention]       │      │")
        diagram.append(f"          │       ({n_heads} heads)         │      │")

        # Add positional encoding info inside attention block
        if pos_enc == "rope":
            diagram.append("          │    (RoPE on Q,K)       │      │")
        elif pos_enc == "alibi":
            diagram.append("          │   (ALiBi bias added)   │      │")
        else:
            diagram.append("          │                         │      │")

        diagram.append("          │ <───────────────────────┘      │")
        diagram.append("          │            +                   │")
        diagram.append("          │                                │")

        # MLP block
        diagram.append(f"          ├──────> [{norm_display}] ─────┐      │")
        diagram.append("          │                         │      │")
        diagram.append("          │         [MLP]           │      │")
        diagram.append(f"          │    ({d_model}→{d_mlp}→{d_model})       │      │")
        diagram.append(f"          │        [{activation_display}]           │      │")
        diagram.append("          │                         │      │")
        diagram.append("          │ <───────────────────────┘      │")
        diagram.append("          │            +                   │")
        diagram.append("          │                                │")

    # Output section
    diagram.append("          │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─      │")
    diagram.append("          │                                │")
    diagram.append(f"          │        [{norm_display}]            │")
    diagram.append("          │                                │")
    diagram.append("          │        [Unembedding]           │")
    diagram.append(f"          │      ({d_model} → vocab)            │")
    diagram.append("          │                                │")
    diagram.append("          │              v                 │")
    diagram.append("          │           OUTPUT               │")
    diagram.append("          │          (logits)              │")
    diagram.append("          └────────────────────────────────┘")

    return "\n".join(diagram)


def generate_graphviz_architecture(config: Dict) -> str:
    """Generate Graphviz DOT code for transformer architecture."""
    n_layers = config.get("n_layers", 4)
    d_model = config.get("d_model", 256)
    n_heads = config.get("n_heads", 4)
    d_mlp = config.get("d_mlp", 1024)
    pos_enc = config.get("positional_encoding", "learned")
    norm = config.get("normalization", "layernorm")
    activation = config.get("activation", "gelu")

    # Start building the DOT code
    dot = []
    dot.append('digraph TransformerArchitecture {')
    dot.append('    bgcolor="black";')
    dot.append('    rankdir=TB;')
    dot.append('    splines=ortho;')  # Use orthogonal lines
    dot.append('    nodesep=0.8;')
    dot.append('    ranksep=0.8;')
    dot.append('    node [shape=box, style=filled, fillcolor="#4a4a4a", fontcolor="white", fontname="Arial", fontsize=11, penwidth=1.5, color="#888888", height=0.5];')
    dot.append('    edge [color="#888888", penwidth=1.5, arrowsize=0.8, fontcolor="white", fontsize=9];')
    dot.append('    ')

    # Input/output nodes
    dot.append('    tokens [label="tokens", fillcolor="#3a3a3a"];')
    dot.append('    embed [label="embed", fillcolor="#5a5a4a"];')

    # Track residual stream positions
    residual_points = []

    # Initial residual point
    dot.append('    x0 [label="x₀", shape=plaintext, fontcolor="white", fontsize=10];')
    residual_points.append('x0')

    # For each layer, create attention heads, MLP, and addition nodes
    for i in range(n_layers):
        layer_id = i + 1

        # Create attention block
        head_label = f"h_{layer_id}"
        if pos_enc == "rope":
            head_label += f"\\n({n_heads} heads, RoPE)"
        elif pos_enc == "alibi":
            head_label += f"\\n({n_heads} heads, ALiBi)"
        else:
            head_label += f"\\n({n_heads} heads)"
        dot.append(f'    h{layer_id} [label="{head_label}", fillcolor="#5a4a5a"];')

        # Addition node after attention
        dot.append(f'    add_attn{layer_id} [label="+", shape=circle, width=0.3, fillcolor="#6a5a4a"];')

        # Residual point after attention
        dot.append(f'    x{layer_id}_mid [label="x_{layer_id}+1", shape=plaintext, fontcolor="white", fontsize=10];')

        # MLP node
        mlp_label = f"MLP m"
        if activation == "swiglu":
            mlp_label = f"MLP m\\n(SwiGLU)"
        elif activation == "gelu":
            mlp_label = f"MLP m\\n(GELU)"
        dot.append(f'    mlp{layer_id} [label="{mlp_label}", fillcolor="#5a5a5a"];')

        # Addition node after MLP
        dot.append(f'    add_mlp{layer_id} [label="+", shape=circle, width=0.3, fillcolor="#6a5a4a"];')

        # Residual point after MLP
        if i < n_layers - 1:
            dot.append(f'    x{layer_id} [label="x_{layer_id}+2", shape=plaintext, fontcolor="white", fontsize=10];')
        else:
            dot.append(f'    x_final [label="x_{n_layers-1}", shape=plaintext, fontcolor="white", fontsize=10];')

    # Output nodes
    dot.append('    unembed [label="unembed", fillcolor="#5a5a4a"];')
    dot.append('    logits [label="logits", fillcolor="#3a3a3a"];')
    dot.append('    ')

    # Add positional embedding if learned
    if pos_enc == "learned":
        dot.append('    pos_embed [label="+PE", shape=box, fillcolor="#4a5a4a"];')

    # Connect everything
    dot.append('    # Connections')
    dot.append('    tokens -> embed;')

    if pos_enc == "learned":
        dot.append('    embed -> pos_embed;')
        dot.append('    pos_embed -> x0;')
    else:
        dot.append('    embed -> x0;')

    # Connect layers
    for i in range(n_layers):
        layer_id = i + 1
        prev_point = 'x0' if i == 0 else f'x{i}'

        # Connect to attention block
        dot.append(f'    {prev_point} -> h{layer_id};')
        dot.append(f'    h{layer_id} -> add_attn{layer_id};')

        # Residual connection to addition
        dot.append(f'    {prev_point} -> add_attn{layer_id} [constraint=false];')

        # From attention to MLP
        dot.append(f'    add_attn{layer_id} -> x{layer_id}_mid;')
        dot.append(f'    x{layer_id}_mid -> mlp{layer_id};')
        dot.append(f'    mlp{layer_id} -> add_mlp{layer_id};')
        dot.append(f'    x{layer_id}_mid -> add_mlp{layer_id} [constraint=false];')

        # To next residual point
        if i < n_layers - 1:
            dot.append(f'    add_mlp{layer_id} -> x{layer_id};')
        else:
            dot.append(f'    add_mlp{n_layers} -> x_final;')

    # Final output
    dot.append('    x_final -> unembed;')
    dot.append('    unembed -> logits;')

    dot.append('}')

    return '\n'.join(dot)


def render_model_architecture_diagram(config: Dict) -> None:
    """Render the model architecture diagram in Streamlit."""
    with st.expander("🏗️ Model Architecture Diagram", expanded=False):
        # Add tabs for different diagram types
        tab1, tab2 = st.tabs(["ASCII Diagram", "Graphviz Diagram"])

        with tab1:
            diagram = generate_model_architecture_diagram(config)
            st.code(diagram, language="text")

            # Add explanation
            st.markdown("""
            **Diagram Legend:**
            - The **Residual Stream** (right side) carries information through the network
            - Components branch off to process information and add it back
            - Each layer has two main blocks: **Attention** and **MLP**
            - Both blocks use normalization and have residual connections (+)
            - The stream preserves dimension d_model throughout the network
            """)

        with tab2:
            try:
                import graphviz
                dot_code = generate_graphviz_architecture(config)
                graph = graphviz.Source(dot_code)
                st.graphviz_chart(dot_code)
            except ImportError:
                st.warning("Graphviz is not installed. Install it with: `pip install graphviz`")
                st.code(generate_graphviz_architecture(config), language="dot")
