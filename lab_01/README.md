# LAB P1-01: Implementação do Mecanismo de Self-Attention

## 1. Objetivo

Este laboratório consiste na implementação da lógica central do mecanismo de **Scaled Dot-Product Attention**, conforme descrito no artigo *"Attention Is All You Need"*. O foco principal é compreender e aplicar a transformação matemática das matrizes de **Query (Q)**, **Key (K)** e **Value (V)**.

## 2. Estrutura do Repositório

```
.
├── lab_01/
│   ├── attention.py       # Implementação da função e da classe de atenção
│   ├── test_attention.py  # Script de testes com exemplos numéricos
│   └── README.md
```

- **`lab_01/attention.py`**: Contém a implementação da função `scaled_dot_product_attention` e da classe `SelfAttentionLayer`, sem utilização de bibliotecas de alto nível de Deep Learning.
- **`lab_01/test_attention.py`**: Script de testes que valida a saída da função com exemplos numéricos simples.

## 3. Como Rodar o Código

**Requisito:** Python 3.10+ com NumPy instalado.

```bash
pip install numpy
```

Execute o script de testes a partir da raiz do repositório:

```bash
python lab_01/test_attention.py
```

## 4. Explicação da Normalização (√d_k)

A implementação baseia-se na fórmula central:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Após o cálculo do produto escalar entre `Q` e `Kᵀ`, o resultado é dividido pela raiz quadrada da dimensão das chaves (`√d_k`), antes da aplicação do Softmax:

```python
scores = Q @ K.T
scaled_scores = scores / np.sqrt(d_k)
```

**Por que isso é necessário?** Quando a dimensão `d_k` é grande, os produtos escalares tendem a crescer em magnitude — pois você está somando `d_k` multiplicações. Valores muito altos empurram o Softmax para regiões de saturação onde o gradiente é extremamente pequeno (*vanishing gradient*), dificultando o aprendizado. Dividir por `√d_k` mantém os scores em uma escala estável.

## 5. Exemplo de Input e Output Esperado

O script `test_attention.py` utiliza o seguinte exemplo para validar o mecanismo:

**Input:**

| Matriz | Valores |
|--------|---------|
| Query (Q) | `[[1.0, 0.0], [0.0, 1.0]]` (Identidade) |
| Key (K) | `[[1.0, 0.0], [0.0, 1.0]]` (Identidade) |
| Value (V) | `[[10.0, 20.0], [30.0, 40.0]]` |

**Output esperado:**

Como Q e K são idênticas, a diagonal de `QKᵀ` é maior que os demais elementos. Após o scaling e o Softmax, os pesos de atenção refletem isso — cada token presta mais atenção em si mesmo do que nos outros.

Pesos de atenção (aproximados):
```
[[0.67, 0.33]
 [0.33, 0.67]]
```

Saída da atenção (aproximada):
```
[[16.6, 26.6]
 [23.4, 33.4]]
```

> **Nota:** O valor maior na diagonal confirma o comportamento correto de self-attention.