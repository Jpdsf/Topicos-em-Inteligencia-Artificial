import numpy as np
from utils.activations import softmax
from utils.projections import linear_project


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)   # [seq_q, seq_k]

    if mask is not None:
        scores = scores + mask

    weights = softmax(scores)  # [seq_q, seq_k]
    output  = weights @ V  # [seq_q, d_v]
    return output, weights


def cross_attention(
    encoder_out: np.ndarray,
    decoder_state: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
) -> np.ndarray:
    batch_size = encoder_out.shape[0]
    outputs = []

    for b in range(batch_size):
        Q = linear_project(decoder_state[b], W_q)  # [seq_dec, d_k]
        K = linear_project(encoder_out[b],   W_k)  # [seq_enc, d_k]
        V = linear_project(encoder_out[b],   W_v)  # [seq_enc, d_v]

        out, _ = scaled_dot_product_attention(Q, K, V, mask=None)
        outputs.append(out)

    return np.stack(outputs, axis=0)  # [batch, seq_dec, d_v]
