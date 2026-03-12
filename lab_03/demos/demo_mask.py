import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from decoder.mask      import create_causal_mask
from decoder.attention import scaled_dot_product_attention


def run():
    print("TAREFA 1 — Máscara Causal (Look-Ahead Mask)")

    seq_len = 5
    d_k     = 8
    np.random.seed(42)

    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)

    mask = create_causal_mask(seq_len)
    print(f"\nMáscara Causal [{seq_len}x{seq_len}]:")
    print(mask)

    _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

    print(f"\nPesos de Atenção após Softmax:")
    print(np.round(weights, 4))

    upper = weights[np.triu(np.ones_like(weights, dtype=bool), k=1)]
    assert np.allclose(upper, 0.0, atol=1e-9), \
        "ERRO: probabilidades futuras não são zero!"

    print("\n Prova Real: todas as posições futuras têm probabilidade 0.0")


if __name__ == "__main__":
    run()
