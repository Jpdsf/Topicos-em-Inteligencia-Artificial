import numpy as np


def layer_norm(x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var  = np.var(x,  axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + epsilon)
