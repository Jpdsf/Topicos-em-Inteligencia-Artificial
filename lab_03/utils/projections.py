import numpy as np


def linear_project(tensor: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    Projeção linear: tensor @ weight.

    Args:
        tensor: [..., d_in]
        weight: [d_in, d_out]

    Returns:
        [..., d_out]
    """
    return tensor @ weight
