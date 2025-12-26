def generate_graphviz_architecture(config):
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
    dot.append('    nodesep=1.0;')
    dot.append('    ranksep=0.6;')
    dot.append('    node [shape=box, style=filled, fillcolor="#4a4a4a", fontcolor="white", fontname="Arial", fontsize=11, penwidth=1.5, color="#888888", height=0.5];')
    dot.append('    edge [color="#888888", penwidth=1.5, arrowsize=0.8];')
    dot.append('    ')

    # Create subgraphs for proper alignment
    dot.append('    # Main residual stream column')
    dot.append('    subgraph cluster_residual {')
    dot.append('        style=invis;')
    dot.append('        ')

    # Input/output in main column
    dot.append('        tokens [label="tokens", fillcolor="#3a3a3a"];')
    dot.append('        embed [label="embed", fillcolor="#5a5a4a"];')

    # Initial residual point
    dot.append('        x0 [label="x₀", shape=point, width=0.15];')

    # Create residual stream points and addition nodes
    for i in range(n_layers):
        layer_id = i + 1
        dot.append(f'        add_attn{layer_id} [label="+", shape=circle, width=0.3, fillcolor="#6a5a4a"];')
        dot.append(f'        x{layer_id}_mid [label="x_{layer_id}+1", shape=point, width=0.15];')
        dot.append(f'        add_mlp{layer_id} [label="+", shape=circle, width=0.3, fillcolor="#6a5a4a"];')

        if i < n_layers - 1:
            dot.append(f'        x{layer_id} [label="x_{layer_id}+2", shape=point, width=0.15];')

    dot.append(f'        x_final [label="x_{n_layers-1}", shape=point, width=0.15];')
    dot.append('        unembed [label="unembed", fillcolor="#5a5a4a"];')
    dot.append('        logits [label="logits", fillcolor="#3a3a3a"];')
    dot.append('    }')
    dot.append('    ')

    # Create attention and MLP blocks to the left
    dot.append('    # Attention and MLP blocks')
    for i in range(n_layers):
        layer_id = i + 1

        # Attention blocks
        if n_heads == 1:
            head_label = f"h_{layer_id}"
        else:
            head_label = f"h₀ h₁ ... h_{n_heads-1}"

        if pos_enc == "rope":
            head_label += "\\n(RoPE)"
        elif pos_enc == "alibi":
            head_label += "\\n(ALiBi)"

        dot.append(f'    h{layer_id} [label="{head_label}", fillcolor="#5a4a5a"];')

        # MLP blocks
        mlp_label = f"MLP m"
        if activation == "swiglu":
            mlp_label += "\\n(SwiGLU)"
        elif activation == "gelu":
            mlp_label += "\\n(GELU)"
        dot.append(f'    mlp{layer_id} [label="{mlp_label}", fillcolor="#5a5a4a"];')

    # Add PE node if needed
    if pos_enc == "learned":
        dot.append('    pos_embed [label="+PE", fillcolor="#4a4a4a"];')

    dot.append('    ')
    dot.append('    # Connections')

    # Main vertical flow
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

        # Vertical residual stream flow
        dot.append(f'    {prev_point} -> add_attn{layer_id};')
        dot.append(f'    add_attn{layer_id} -> x{layer_id}_mid;')
        dot.append(f'    x{layer_id}_mid -> add_mlp{layer_id};')

        if i < n_layers - 1:
            dot.append(f'    add_mlp{layer_id} -> x{layer_id};')
        else:
            dot.append(f'    add_mlp{layer_id} -> x_final;')

        # Attention branches off to the left
        dot.append(f'    {prev_point} -> h{layer_id} [constraint=false];')
        dot.append(f'    h{layer_id} -> add_attn{layer_id};')

        # MLP branches off to the left
        dot.append(f'    x{layer_id}_mid -> mlp{layer_id} [constraint=false];')
        dot.append(f'    mlp{layer_id} -> add_mlp{layer_id};')

    # Final output
    dot.append('    x_final -> unembed;')
    dot.append('    unembed -> logits;')

    # Force horizontal alignment of components
    for i in range(n_layers):
        layer_id = i + 1
        dot.append(f'    {{rank=same; h{layer_id}; add_attn{layer_id};}}')
        dot.append(f'    {{rank=same; mlp{layer_id}; add_mlp{layer_id};}}')

    dot.append('}')

    return '\n'.join(dot)


# Test the function
if __name__ == "__main__":
    config = {
        "n_layers": 2,
        "d_model": 512,
        "n_heads": 8,
        "d_mlp": 2048,
        "positional_encoding": "rope",
        "normalization": "layernorm",
        "activation": "gelu"
    }

    dot_code = generate_graphviz_architecture(config)
    print(dot_code)