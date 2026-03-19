import numpy as np
from .encoder_block import EncoderBlock


class Encoder:
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int = 2,
    ) -> None:
        self.layers = [
            EncoderBlock(d_model, num_heads, d_ff, seed=42 + i)
            for i in range(num_layers)
        ]

    def __call__(
        self,
        x: np.ndarray,
        src_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        for layer in self.layers:
            x = layer(x, mask=src_mask)
        return x
