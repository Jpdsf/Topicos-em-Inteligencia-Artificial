import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax numericamente estável.

    Subtrai o máximo antes de exponenciar para evitar overflow/underflow.

    Args:
        x:    Array de entrada (qualquer shape).
        axis: Eixo ao longo do qual normalizar (default: último).

    Returns:
        Array com mesma shape de x, valores somando 1 ao longo de `axis`.
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)
