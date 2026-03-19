import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from transformer import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    PositionWiseFFN,
    add_and_norm,
    positional_encoding,
    make_causal_mask,
    EncoderBlock,
    Encoder,
    DecoderBlock,
    Decoder,
)

BATCH, SEQ, D_MODEL, NUM_HEADS, D_FF = 2, 5, 32, 4, 128
rng = np.random.default_rng(0)


@pytest.fixture
def sample_tensor():
    return rng.normal(size=(BATCH, SEQ, D_MODEL))


class TestScaledDotProductAttention:
    def test_output_shape(self, sample_tensor):
        Q = K = V = sample_tensor
        out, weights = scaled_dot_product_attention(Q, K, V)
        assert out.shape == (BATCH, SEQ, D_MODEL)
        assert weights.shape == (BATCH, SEQ, SEQ)

    def test_weights_sum_to_one(self, sample_tensor):
        Q = K = V = sample_tensor
        _, weights = scaled_dot_product_attention(Q, K, V)
        sums = weights.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_causal_mask_zeros_upper_triangle(self):
        seq = 4
        x = rng.normal(size=(1, seq, D_MODEL))
        mask = make_causal_mask(seq)
        _, weights = scaled_dot_product_attention(x, x, x, mask=mask)
        upper = np.triu(np.ones((seq, seq)), k=1).astype(bool)
        assert np.allclose(weights[0, 0][upper], 0.0, atol=1e-6)


class TestPositionWiseFFN:
    def test_output_shape(self, sample_tensor):
        ffn = PositionWiseFFN(D_MODEL, D_FF)
        out = ffn(sample_tensor)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_different_positions_independent(self, sample_tensor):
        ffn = PositionWiseFFN(D_MODEL, D_FF)
        out_full = ffn(sample_tensor)
        out_pos0 = ffn(sample_tensor[:, :1, :])
        assert np.allclose(out_full[:, :1, :], out_pos0, atol=1e-6)


class TestAddAndNorm:
    def test_output_shape(self, sample_tensor):
        out = add_and_norm(sample_tensor, sample_tensor)
        assert out.shape == sample_tensor.shape

    def test_normalized(self, sample_tensor):
        sublayer = rng.normal(size=sample_tensor.shape)
        out = add_and_norm(sample_tensor, sublayer)
        assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-5)
        assert np.allclose(out.var(axis=-1), 1.0, atol=1e-5)


class TestPositionalEncoding:
    def test_shape(self):
        pe = positional_encoding(SEQ, D_MODEL)
        assert pe.shape == (1, SEQ, D_MODEL)

    def test_deterministic(self):
        pe1 = positional_encoding(SEQ, D_MODEL)
        pe2 = positional_encoding(SEQ, D_MODEL)
        assert np.array_equal(pe1, pe2)


class TestCausalMask:
    def test_shape(self):
        mask = make_causal_mask(SEQ)
        assert mask.shape == (1, 1, SEQ, SEQ)

    def test_upper_triangle_is_neg_inf(self):
        mask = make_causal_mask(SEQ)
        upper = np.triu(np.ones((SEQ, SEQ)), k=1).astype(bool)
        assert np.all(np.isinf(mask[0, 0][upper]))
        assert np.all(mask[0, 0][upper] < 0)

    def test_lower_triangle_is_zero(self):
        mask = make_causal_mask(SEQ)
        lower = ~np.triu(np.ones((SEQ, SEQ)), k=1).astype(bool)
        assert np.all(mask[0, 0][lower] == 0.0)


class TestEncoder:
    def test_encoder_block_shape(self, sample_tensor):
        block = EncoderBlock(D_MODEL, NUM_HEADS, D_FF)
        out = block(sample_tensor)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_encoder_stack_shape(self, sample_tensor):
        enc = Encoder(D_MODEL, NUM_HEADS, D_FF, num_layers=2)
        Z = enc(sample_tensor)
        assert Z.shape == (BATCH, SEQ, D_MODEL)

    def test_encoder_output_differs_from_input(self, sample_tensor):
        enc = Encoder(D_MODEL, NUM_HEADS, D_FF)
        Z = enc(sample_tensor)
        assert not np.allclose(Z, sample_tensor)


class TestDecoder:
    def test_decoder_block_shape(self, sample_tensor):
        Z   = rng.normal(size=(BATCH, SEQ, D_MODEL))
        blk = DecoderBlock(D_MODEL, NUM_HEADS, D_FF)
        out = blk(sample_tensor, Z)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_decoder_output_shape(self, sample_tensor):
        VOCAB_SIZE = 20
        Z   = rng.normal(size=(BATCH, SEQ, D_MODEL))
        dec = Decoder(D_MODEL, NUM_HEADS, D_FF, VOCAB_SIZE, num_layers=2)
        probs = dec(sample_tensor, Z)
        assert probs.shape == (BATCH, SEQ, VOCAB_SIZE)

    def test_decoder_probs_sum_to_one(self, sample_tensor):
        VOCAB_SIZE = 20
        Z   = rng.normal(size=(BATCH, SEQ, D_MODEL))
        dec = Decoder(D_MODEL, NUM_HEADS, D_FF, VOCAB_SIZE, num_layers=2)
        probs = dec(sample_tensor, Z)
        sums = probs.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_cross_attention_uses_encoder_memory(self, sample_tensor):
        VOCAB_SIZE = 20
        Z1  = rng.normal(size=(BATCH, SEQ, D_MODEL))
        Z2  = rng.normal(size=(BATCH, SEQ, D_MODEL)) * 10
        dec = Decoder(D_MODEL, NUM_HEADS, D_FF, VOCAB_SIZE, num_layers=2)
        p1  = dec(sample_tensor, Z1)
        p2  = dec(sample_tensor, Z2)
        assert not np.allclose(p1, p2)
