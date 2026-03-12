import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from decoder.mask      import create_causal_mask
from decoder.attention import scaled_dot_product_attention


def test_mask_shape():
    for n in [1, 3, 5, 10]:
        mask = create_causal_mask(n)
        assert mask.shape == (n, n), f"Shape incorreto para n={n}: {mask.shape}"
    print("  [OK] test_mask_shape")


def test_mask_lower_triangle_is_zero():
    mask = create_causal_mask(5)
    rows, cols = np.tril_indices(5)
    assert np.all(mask[rows, cols] == 0.0), "Triangular inferior contém valores != 0"
    print("  [OK] test_mask_lower_triangle_is_zero")


def test_mask_upper_triangle_is_neginf():
    mask = create_causal_mask(5)
    rows, cols = np.triu_indices(5, k=1)
    assert np.all(np.isneginf(mask[rows, cols])), "Triangular superior não é -inf"
    print("  [OK] test_mask_upper_triangle_is_neginf")


def test_softmax_future_probs_are_zero():
    np.random.seed(99)
    seq_len, d_k = 6, 16
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)
    mask = create_causal_mask(seq_len)

    _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    upper = weights[np.triu(np.ones_like(weights, dtype=bool), k=1)]
    assert np.allclose(upper, 0.0, atol=1e-9), "Há probabilidades > 0 no futuro"
    print("  [OK] test_softmax_future_probs_are_zero")


def run_all():
    print("\n── Testes: Máscara Causal ──")
    test_mask_shape()
    test_mask_lower_triangle_is_zero()
    test_mask_upper_triangle_is_neginf()
    test_softmax_future_probs_are_zero()
    print("  Todos os testes passaram!\n")


if __name__ == "__main__":
    run_all()
