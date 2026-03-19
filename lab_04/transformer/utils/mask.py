import numpy as np


def make_causal_mask(seq_len: int) -> np.ndarray:
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
    return mask[np.newaxis, np.newaxis, :, :]
