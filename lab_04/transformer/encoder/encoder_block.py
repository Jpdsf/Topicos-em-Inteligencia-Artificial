import numpy as np
from transformer.attention import MultiHeadAttention
from transformer.utils import PositionWiseFFN, add_and_norm


class EncoderBlock:
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        seed: int = 42,
    ) -> None:
        self.self_attention = MultiHeadAttention(d_model, num_heads, seed=seed)
        self.ffn            = PositionWiseFFN(d_model, d_ff, seed=seed + 1)

    def __call__(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        attn_out = self.self_attention(x, x, x, mask=mask)
        x = add_and_norm(x, attn_out)

        ffn_out = self.ffn(x)
        x = add_and_norm(x, ffn_out)

        return x
