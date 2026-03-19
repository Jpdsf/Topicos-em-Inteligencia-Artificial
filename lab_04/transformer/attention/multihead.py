import numpy as np
from .scaled_dot_product import scaled_dot_product_attention


class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int, seed: int = 42) -> None:
        assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / d_model)

        self.W_Q = rng.normal(0, scale, (d_model, d_model))
        self.W_K = rng.normal(0, scale, (d_model, d_model))
        self.W_V = rng.normal(0, scale, (d_model, d_model))
        self.W_O = rng.normal(0, scale, (d_model, d_model))

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch, seq, _ = x.shape
        x = x.reshape(batch, seq, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        batch, _, seq, _ = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch, seq, self.d_model)

    def __call__(
        self,
        Q_in: np.ndarray,
        K_in: np.ndarray,
        V_in: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        Q = Q_in @ self.W_Q
        K = K_in @ self.W_K
        V = V_in @ self.W_V

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)

        concat = self._merge_heads(attn_output)
        return concat @ self.W_O
