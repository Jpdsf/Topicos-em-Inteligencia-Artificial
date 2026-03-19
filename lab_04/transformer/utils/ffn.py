import numpy as np


class PositionWiseFFN:
    def __init__(self, d_model: int, d_ff: int, seed: int = 42) -> None:
        self.d_model = d_model
        self.d_ff = d_ff

        rng = np.random.default_rng(seed)
        scale_1 = np.sqrt(2.0 / d_model)
        scale_2 = np.sqrt(2.0 / d_ff)

        self.W_1 = rng.normal(0, scale_1, (d_model, d_ff))
        self.b_1 = np.zeros(d_ff)
        self.W_2 = rng.normal(0, scale_2, (d_ff, d_model))
        self.b_2 = np.zeros(d_model)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0, x @ self.W_1 + self.b_1)
        return hidden @ self.W_2 + self.b_2
