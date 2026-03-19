import numpy as np
from transformer.attention import MultiHeadAttention
from transformer.utils import PositionWiseFFN, add_and_norm


class DecoderBlock:
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        seed: int = 100,
    ) -> None:
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads, seed=seed)
        self.cross_attention        = MultiHeadAttention(d_model, num_heads, seed=seed + 1)
        self.ffn                    = PositionWiseFFN(d_model, d_ff, seed=seed + 2)

    def __call__(
        self,
        y: np.ndarray,
        Z: np.ndarray,
        causal_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        masked_attn = self.masked_self_attention(y, y, y, mask=causal_mask)
        y = add_and_norm(y, masked_attn)

        cross_attn = self.cross_attention(y, Z, Z, mask=None)
        y = add_and_norm(y, cross_attn)

        ffn_out = self.ffn(y)
        y = add_and_norm(y, ffn_out)

        return y
