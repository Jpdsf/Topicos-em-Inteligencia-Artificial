import numpy as np
from transformer.layers.attention import ScaledDotProductAttention
from transformer.layers.feed_forward import FeedForwardNetwork
from transformer.utils.normalization import layer_norm


class EncoderLayer:

    def __init__(self, d_model: int, d_ff: int, epsilon: float = 1e-6) -> None:
        self.attention = ScaledDotProductAttention(d_model)
        self.ffn       = FeedForwardNetwork(d_model, d_ff)
        self.epsilon   = epsilon

    def _add_and_norm(self, x: np.ndarray, sublayer_out: np.ndarray) -> np.ndarray:
        return layer_norm(x + sublayer_out, epsilon=self.epsilon)

    def forward(self, X: np.ndarray) -> np.ndarray:
        
        X_att   = self.attention.forward(X)
        X_norm1 = self._add_and_norm(X, X_att)

        X_ffn = self.ffn.forward(X_norm1)
        X_out = self._add_and_norm(X_norm1, X_ffn)

        return X_out
