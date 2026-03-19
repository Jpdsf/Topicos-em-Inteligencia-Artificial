import numpy as np
from .decoder_block import DecoderBlock
from transformer.utils import make_causal_mask


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)


class Decoder:
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        num_layers: int = 2,
    ) -> None:
        self.d_model    = d_model
        self.vocab_size = vocab_size

        self.layers = [
            DecoderBlock(d_model, num_heads, d_ff, seed=100 + i * 10)
            for i in range(num_layers)
        ]

        rng   = np.random.default_rng(999)
        scale = np.sqrt(2.0 / d_model)
        self.W_out = rng.normal(0, scale, (d_model, vocab_size))
        self.b_out = np.zeros(vocab_size)

    def __call__(
        self,
        y: np.ndarray,
        Z: np.ndarray,
    ) -> np.ndarray:
        tgt_seq   = y.shape[1]
        causal_mask = make_causal_mask(tgt_seq)

        for layer in self.layers:
            y = layer(y, Z, causal_mask=causal_mask)

        logits = y @ self.W_out + self.b_out
        return _softmax(logits)
