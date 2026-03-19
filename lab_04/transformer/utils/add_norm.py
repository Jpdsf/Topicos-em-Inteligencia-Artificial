import numpy as np


def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray | None = None,
    beta: np.ndarray | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)

    if gamma is not None:
        x_norm = x_norm * gamma
    if beta is not None:
        x_norm = x_norm + beta

    return x_norm


def add_and_norm(
    x: np.ndarray,
    sublayer_output: np.ndarray,
    gamma: np.ndarray | None = None,
    beta: np.ndarray | None = None,
) -> np.ndarray:
    return layer_norm(x + sublayer_output, gamma=gamma, beta=beta)
