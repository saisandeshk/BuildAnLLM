
import torch

def extend_positional_embeddings(pos_embed_module, new_max_length: int):
    """
    Extend positional embeddings to support longer sequences.
    
    Uses interpolation to extend the embedding matrix. This is a common
    technique for extending pre-trained positional embeddings to longer contexts.
    
    Args:
        pos_embed_module: The positional embedding module (PosEmbedWithEinops or PosEmbedWithoutEinops)
        new_max_length: New maximum sequence length
    """
    old_W_pos = pos_embed_module.W_pos  # [old_n_ctx, d_model]
    old_n_ctx, d_model = old_W_pos.shape

    if new_max_length <= old_n_ctx:
        # No extension needed
        return

    # Create new embedding matrix
    new_W_pos = torch.empty((new_max_length, d_model),
                            device=old_W_pos.device, dtype=old_W_pos.dtype)

    # Copy existing embeddings
    new_W_pos[:old_n_ctx] = old_W_pos

    # For positions beyond the original length, use interpolation
    # Method: Use the last few positions to extrapolate smoothly
    if old_n_ctx >= 2:
        # Use the trend from the last few positions
        # Compute average "velocity" (difference between consecutive positions)
        # and extrapolate
        last_few = min(10, old_n_ctx)  # Use last 10 positions or all if fewer
        recent_embeds = old_W_pos[-last_few:]  # [last_few, d_model]

        # Compute average change per position
        if last_few >= 2:
            diffs = recent_embeds[1:] - \
                recent_embeds[:-1]  # [last_few-1, d_model]
            # [d_model] - average change per position
            avg_diff = diffs.mean(dim=0)
        else:
            avg_diff = torch.zeros_like(old_W_pos[-1])

        # Extrapolate: start from last position and add scaled differences
        last_embed = old_W_pos[-1]  # [d_model]
        for i in range(old_n_ctx, new_max_length):
            # Scale the difference based on how far we are from the original range
            # Use a decay factor to prevent embeddings from growing too large
            steps_from_end = i - old_n_ctx + 1
            decay = 0.9 ** steps_from_end  # Exponential decay
            new_W_pos[i] = last_embed + avg_diff * steps_from_end * decay
    else:
        # If only one position, just repeat it
        new_W_pos[old_n_ctx:] = old_W_pos[-1]

    # Update the parameter
    pos_embed_module.W_pos = torch.nn.Parameter(new_W_pos)
    # Update cfg.n_ctx to reflect the new max length
    pos_embed_module.cfg.n_ctx = new_max_length
