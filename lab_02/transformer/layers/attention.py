import numpy as np
from transformer.utils.activations import softmax


class ScaledDotProductAttention:

    def __init__(self, d_model: int) -> None:
        self.d_k = d_model

        scale = np.sqrt(2.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale

    def _project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Q = X @ self.W_Q   
        K = X @ self.W_K 
        V = X @ self.W_V 
        return Q, K, V

    def _compute_scores(self, Q: np.ndarray, K: np.ndarray) -> np.ndarray:
        scores = Q @ K.transpose(0, 2, 1) 
        return scores / np.sqrt(self.d_k)

    def forward(self, X: np.ndarray) -> np.ndarray:
        
        Q, K, V          = self._project(X)
        scores           = self._compute_scores(Q, K)
        attention_weights = softmax(scores, axis=-1)   
        output           = attention_weights @ V  
        return output
