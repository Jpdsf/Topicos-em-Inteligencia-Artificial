import numpy as np
from attention import SelfAttentionLayer, scaled_dot_product_attention

def test_numerical_example():
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([[10.0, 20.0], [30.0, 40.0]])
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print("Pesos de atenção calculados:\n", weights)
    print("Saída da atenção:\n", output)
    
    assert weights[0, 0] > weights[0, 1], "A auto-atenção deveria ser maior na diagonal"
    print("Teste numérico aprovado!")

if __name__ == "__main__":
    try:
        test_numerical_example()
        print("\nTodos os testes passaram com sucesso!")
    except AssertionError as e:
        print(f"\nFalha nos testes: {e}")