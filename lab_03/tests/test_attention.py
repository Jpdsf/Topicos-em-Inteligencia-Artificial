import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from decoder.attention import scaled_dot_product_attention, cross_attention


def test_attention_output_shape():
    Q = np.random.randn(4, 8)
    K = np.random.randn(6, 8)
    V = np.random.randn(6, 16)
    out, w = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (4, 16), f"Shape incorreto: {out.shape}"
    assert w.shape   == (4, 6),  f"Weights shape incorreto: {w.shape}"
    print("  [OK] test_attention_output_shape")


def test_attention_weights_sum_to_one():
    Q = np.random.randn(5, 8)
    K = np.random.randn(5, 8)
    V = np.random.randn(5, 8)
    _, w = scaled_dot_product_attention(Q, K, V)
    row_sums = w.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"Linhas não somam 1: {row_sums}"
    print("  [OK] test_attention_weights_sum_to_one")


def test_cross_attention_output_shape():
    np.random.seed(7)
    batch, seq_enc, seq_dec, d_model, d_k, d_v = 2, 10, 4, 64, 16, 16
    enc = np.random.randn(batch, seq_enc, d_model)
    dec = np.random.randn(batch, seq_dec, d_model)
    W_q = np.random.randn(d_model, d_k)
    W_k = np.random.randn(d_model, d_k)
    W_v = np.random.randn(d_model, d_v)
    out = cross_attention(enc, dec, W_q, W_k, W_v)
    assert out.shape == (batch, seq_dec, d_v), f"Shape incorreto: {out.shape}"
    print("  [OK] test_cross_attention_output_shape")


def run_all():
    print("\n── Testes: Attention ──")
    test_attention_output_shape()
    test_attention_weights_sum_to_one()
    test_cross_attention_output_shape()
    print("  Todos os testes passaram!\n")


if __name__ == "__main__":
    run_all()
