"""Unit tests for positional embedding classes."""

import pytest
import torch
from pretraining.positional_embeddings.positional_embedding import PosEmbed
from pretraining.positional_embeddings.rope import RoPE
from pretraining.positional_embeddings.alibi import ALiBi


@pytest.mark.unit
@pytest.mark.parametrize("use_einops", [True, False])
class TestPosEmbed:
    """Tests for PosEmbed (learned positional embeddings)."""

    def test_init(self, gpt_config, use_einops):
        """Test initialization."""
        pos_embed = PosEmbed(gpt_config, use_einops=use_einops)
        assert pos_embed.W_pos.shape == (gpt_config.n_ctx, gpt_config.d_model)
        assert isinstance(pos_embed.W_pos, torch.nn.Parameter)

    def test_forward(self, gpt_config, sample_tokens, use_einops):
        """Test forward pass."""
        pos_embed = PosEmbed(gpt_config, use_einops=use_einops)
        output = pos_embed(sample_tokens)
        batch_size, seq_len = sample_tokens.shape
        assert output.shape == (batch_size, seq_len, gpt_config.d_model)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_broadcast_embeddings(self, gpt_config, use_einops):
        """Test broadcasting to batch dimension."""
        pos_embed = PosEmbed(gpt_config, use_einops=use_einops)
        seq_len = 5
        position_embeddings = torch.randn(seq_len, gpt_config.d_model)
        batch_size = 3
        broadcasted = pos_embed._broadcast_embeddings(position_embeddings, batch_size)
        assert broadcasted.shape == (batch_size, seq_len, gpt_config.d_model)
        # Check that all batches have same embeddings
        assert torch.allclose(broadcasted[0], broadcasted[1])
        assert torch.allclose(broadcasted[0], broadcasted[2])

    def test_different_sequence_lengths(self, gpt_config, use_einops):
        """Test with different sequence lengths."""
        pos_embed = PosEmbed(gpt_config, use_einops=use_einops)
        for seq_len in [1, 5, 10, gpt_config.n_ctx]:
            tokens = torch.randint(0, gpt_config.d_vocab, (2, seq_len))
            output = pos_embed(tokens)
            assert output.shape == (2, seq_len, gpt_config.d_model)

@pytest.mark.unit
class TestPosEmbedEquivalence:
    """Tests for PosEmbed equivalence between implementations."""

    def test_equivalence_einops_vs_torch(self, gpt_config, sample_tokens):
        """Test that einops and non-einops implementations are equivalent."""
        pos_embed1 = PosEmbed(gpt_config, use_einops=True)
        pos_embed2 = PosEmbed(gpt_config, use_einops=False)
        
        # Copy parameters
        pos_embed2.W_pos.data = pos_embed1.W_pos.data.clone()
        
        output1 = pos_embed1(sample_tokens)
        output2 = pos_embed2(sample_tokens)
        assert torch.allclose(output1, output2, atol=1e-5)

    @pytest.mark.parametrize("use_einops", [True, False])
    def test_gradient_flow(self, gpt_config, sample_tokens, use_einops):
        """Test that gradients flow through positional embeddings."""
        pos_embed = PosEmbed(gpt_config, use_einops=use_einops)
        output = pos_embed(sample_tokens)
        loss = output.sum()
        loss.backward()
        assert pos_embed.W_pos.grad is not None


@pytest.mark.unit
class TestRoPE:
    """Tests for RoPE (Rotary Position Embedding)."""

    def test_init(self, llama_config):
        """Test initialization."""
        rope = RoPE(llama_config)
        assert rope.d_head == llama_config.d_head
        assert rope.theta == llama_config.rope_theta
        assert hasattr(rope, 'freqs')
        assert rope.freqs.shape == (llama_config.d_head // 2,)

    def test_compute_rotation_angles(self, llama_config):
        """Test rotation angle computation."""
        rope = RoPE(llama_config)
        positions = torch.arange(5)
        cos, sin = rope._compute_rotation_angles(positions)
        assert cos.shape == (5, llama_config.d_head // 2)
        assert sin.shape == (5, llama_config.d_head // 2)
        # Check that cos^2 + sin^2 = 1
        assert torch.allclose(cos ** 2 + sin ** 2, torch.ones_like(cos), atol=1e-5)

    def test_reshape_to_pairs(self, llama_config):
        """Test reshaping to pairs."""
        rope = RoPE(llama_config)
        batch, seq, n_heads, d_head = 2, 5, llama_config.n_heads, llama_config.d_head
        q = torch.randn(batch, seq, n_heads, d_head)
        pairs = rope._reshape_to_pairs(q)
        assert pairs.shape == (batch, seq, n_heads, d_head // 2, 2)

    def test_reshape_from_pairs(self, llama_config):
        """Test reshaping from pairs."""
        rope = RoPE(llama_config)
        batch, seq, n_heads, d_head = 2, 5, llama_config.n_heads, llama_config.d_head
        pairs = torch.randn(batch, seq, n_heads, d_head // 2, 2)
        original = rope._reshape_from_pairs(pairs, (batch, seq, n_heads, d_head))
        assert original.shape == (batch, seq, n_heads, d_head)

    def test_apply_rotation(self, llama_config):
        """Test rotation application."""
        rope = RoPE(llama_config)
        batch, seq, n_heads, d_head = 2, 5, llama_config.n_heads, llama_config.d_head
        pairs = torch.randn(batch, seq, n_heads, d_head // 2, 2)
        positions = torch.arange(seq)
        cos, sin = rope._compute_rotation_angles(positions)
        rotated = rope._apply_rotation(pairs, cos, sin)
        assert rotated.shape == pairs.shape

    def test_forward(self, llama_config):
        """Test forward pass."""
        rope = RoPE(llama_config)
        batch, seq, n_heads, d_head = 2, 5, llama_config.n_heads, llama_config.d_head
        q = torch.randn(batch, seq, n_heads, d_head)
        k = torch.randn(batch, seq, llama_config.n_kv_heads, d_head)
        positions = torch.arange(seq)
        
        q_rotated, k_rotated = rope(q, k, positions)
        assert q_rotated.shape == q.shape
        assert k_rotated.shape == k.shape

    def test_forward_gqa(self, gqa_config):
        """Test forward pass with GQA (different n_kv_heads)."""
        rope = RoPE(gqa_config)
        batch, seq, n_heads, d_head = 2, 5, gqa_config.n_heads, gqa_config.d_head
        q = torch.randn(batch, seq, n_heads, d_head)
        k = torch.randn(batch, seq, gqa_config.n_kv_heads, d_head)
        positions = torch.arange(seq)
        
        q_rotated, k_rotated = rope(q, k, positions)
        assert q_rotated.shape == q.shape
        assert k_rotated.shape == k.shape

    def test_rotation_properties(self, llama_config):
        """Test that rotation preserves magnitude."""
        rope = RoPE(llama_config)
        batch, seq, n_heads, d_head = 1, 3, llama_config.n_heads, llama_config.d_head
        q = torch.randn(batch, seq, n_heads, d_head)
        k = torch.randn(batch, seq, llama_config.n_kv_heads, d_head)
        positions = torch.arange(seq)
        
        q_orig_norm = torch.norm(q, dim=-1)
        k_orig_norm = torch.norm(k, dim=-1)
        
        q_rotated, k_rotated = rope(q, k, positions)
        
        q_rot_norm = torch.norm(q_rotated, dim=-1)
        k_rot_norm = torch.norm(k_rotated, dim=-1)
        
        # Rotation should preserve magnitude
        assert torch.allclose(q_orig_norm, q_rot_norm, atol=1e-5)
        assert torch.allclose(k_orig_norm, k_rot_norm, atol=1e-5)

    def test_different_positions(self, llama_config):
        """Test that different positions produce different rotations."""
        rope = RoPE(llama_config)
        batch, n_heads, d_head = 1, llama_config.n_heads, llama_config.d_head
        q1 = torch.randn(batch, 1, n_heads, d_head)
        q2 = torch.randn(batch, 1, n_heads, d_head)
        # Use same values
        q2.data = q1.data.clone()
        
        positions1 = torch.tensor([0])
        positions2 = torch.tensor([1])
        k_dummy = torch.randn(batch, 1, llama_config.n_kv_heads, d_head)
        
        q1_rot, _ = rope(q1, k_dummy, positions1)
        q2_rot, _ = rope(q2, k_dummy, positions2)
        
        # Different positions should produce different rotations
        assert not torch.allclose(q1_rot, q2_rot, atol=1e-5)


@pytest.mark.unit
class TestALiBi:
    """Tests for ALiBi (Attention with Linear Biases)."""

    def test_init(self, olmo_config):
        """Test initialization."""
        alibi = ALiBi(olmo_config)
        assert alibi.n_heads == olmo_config.n_heads
        assert hasattr(alibi, 'slopes')
        assert alibi.slopes.shape == (olmo_config.n_heads,)
        # Slopes should be decreasing
        assert torch.all(alibi.slopes[:-1] >= alibi.slopes[1:])

    def test_compute_distance_matrix(self, olmo_config, device):
        """Test distance matrix computation."""
        alibi = ALiBi(olmo_config)
        seq_len = 5
        distance = alibi._compute_distance_matrix(seq_len, device)
        assert distance.shape == (seq_len, seq_len)
        # Check symmetry
        assert torch.allclose(distance, distance.t())
        # Check diagonal is 0
        assert torch.allclose(torch.diag(distance), torch.zeros(seq_len))
        # Check distance[i, j] = |i - j|
        for i in range(seq_len):
            for j in range(seq_len):
                assert abs(distance[i, j].item() - abs(i - j)) < 1e-5

    def test_apply_slopes(self, olmo_config, device):
        """Test slope application."""
        alibi = ALiBi(olmo_config)
        seq_len = 5
        distance = alibi._compute_distance_matrix(seq_len, device)
        bias = alibi._apply_slopes(distance)
        assert bias.shape == (olmo_config.n_heads, seq_len, seq_len)
        # Bias should be negative (slopes are positive, distance is non-negative)
        assert torch.all(bias <= 0)

    def test_apply_causal_mask(self, olmo_config, device):
        """Test causal masking."""
        alibi = ALiBi(olmo_config)
        seq_len = 5
        distance = alibi._compute_distance_matrix(seq_len, device)
        bias = alibi._apply_slopes(distance)
        masked_bias = alibi._apply_causal_mask(bias, seq_len, device)

        assert torch.allclose(masked_bias, bias, atol=1e-6)

        # Diagonal should be zero, off-diagonal should be negative.
        for i in range(seq_len):
            assert torch.allclose(masked_bias[:, i, i], torch.zeros(olmo_config.n_heads), atol=1e-5)
            for j in range(seq_len):
                if i != j:
                    assert torch.all(masked_bias[:, i, j] < 0)

    def test_get_bias(self, olmo_config, device):
        """Test get_bias method."""
        alibi = ALiBi(olmo_config)
        seq_len = 5
        bias = alibi.get_bias(seq_len, device)
        assert bias.shape == (olmo_config.n_heads, seq_len, seq_len)
        
        # Diagonal should be zero, off-diagonal should be negative.
        for i in range(seq_len):
            assert torch.allclose(bias[:, i, i], torch.zeros(olmo_config.n_heads), atol=1e-5)
            for j in range(seq_len):
                if i != j:
                    assert torch.all(bias[:, i, j] < 0)

    def test_different_sequence_lengths(self, olmo_config, device):
        """Test with different sequence lengths."""
        alibi = ALiBi(olmo_config)
        for seq_len in [1, 5, 10, 20]:
            bias = alibi.get_bias(seq_len, device)
            assert bias.shape == (olmo_config.n_heads, seq_len, seq_len)

    def test_bias_monotonicity(self, olmo_config, device):
        """Test that bias becomes more negative with distance."""
        alibi = ALiBi(olmo_config)
        seq_len = 10
        bias = alibi.get_bias(seq_len, device)
        
        # For a fixed position i, bias should become more negative as j increases (for j > i)
        i = 2
        for j in range(i + 1, seq_len - 1):
            assert torch.all(bias[:, i, j] >= bias[:, i, j + 1])

    def test_per_head_differences(self, olmo_config, device):
        """Test that different heads have different slopes."""
        alibi = ALiBi(olmo_config)
        seq_len = 5
        bias = alibi.get_bias(seq_len, device)
        
        # Different heads should have different biases (unless slopes are identical, which they shouldn't be)
        if olmo_config.n_heads > 1:
            # Check that at least some heads differ
            head0_bias = bias[0]  # [seq_len, seq_len]
            head1_bias = bias[1]  # [seq_len, seq_len]
            # They should differ for future positions (first row, positions after 0)
            assert not torch.allclose(head0_bias[0, 1:], head1_bias[0, 1:], atol=1e-5)
