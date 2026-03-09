import numpy as np
from transformer.utils.activations import relu


class FeedForwardNetwork:

    def __init__(self, d_model: int, d_ff: int) -> None:
        scale = np.sqrt(2.0 / d_model)

        self.W1 = np.random.randn(d_model, d_ff) * scale
        self.b1 = np.zeros(d_ff)

        self.W2 = np.random.randn(d_ff, d_model) * scale
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = relu(x @ self.W1 + self.b1) 
        output = hidden @ self.W2 + self.b2   
        return output
