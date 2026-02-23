import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    d_k = Q.shape[-1]

    scores = Q @ K.T
    
    scaled_scores = scores / np.sqrt(d_k)
    
    attention_weights = softmax(scaled_scores)
    
    output = attention_weights @ V
    
    return output, attention_weights

class SelfAttentionLayer:
    def __init__(self, dimensao_modelo: int):
        self.d_k = dimensao_modelo
        self.W_q = np.random.randn(dimensao_modelo, dimensao_modelo) * 0.1
        self.W_k = np.random.randn(dimensao_modelo, dimensao_modelo) * 0.1
        self.W_v = np.random.randn(dimensao_modelo, dimensao_modelo) * 0.1

    def forward(self, x: np.ndarray):
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        return scaled_dot_product_attention(Q, K, V)