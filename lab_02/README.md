# Transformer Encoder — From Scratch

> **Disciplina:** Tópicos em Inteligência Artificial – 2026.1  
> **Professor:** Prof. Dimmy Magalhães  
> **Instituição:** iCEV – Instituto de Ensino Superior

Implementação do *Forward Pass* completo de um **Encoder Transformer**,
conforme Vaswani et al. (2017) — *"Attention Is All You Need"* —
utilizando exclusivamente `Python 3`, `NumPy` e `pandas`.

---

## Estrutura do Projeto

```
transformer_encoder/
│
├── config.py              # Hiperparâmetros centralizados (dataclass imutável)
├── data_pipeline.py       # Vocabulário, tokenização, embeddings, tensor X
├── main.py                # Ponto de entrada — orquestra o pipeline completo
│
├── transformer/
│   ├── encoder.py         # TransformerEncoder (pilha de N camadas)
│   ├── layers/
│   │   ├── attention.py       # Scaled Dot-Product Attention
│   │   ├── feed_forward.py    # Feed-Forward Network (FFN)
│   │   └── encoder_layer.py   # EncoderLayer (Attention + FFN + Add & Norm)
│   └── utils/
│       ├── activations.py     # softmax, relu
│       └── normalization.py   # layer_norm
│
└── tests/
    └── test_transformer.py    # 21 testes unitários (pytest-compatível)
```

---

## Arquitetura

```
Frase de texto
      │
      ▼
DataPipeline
  ├─ Vocabulário (pandas DataFrame)
  ├─ Tokenização  →  [id₀, id₁, …, idₙ]
  └─ Embedding Table (vocab_size × d_model)
      │
      ▼
Tensor X : (batch=1, seq_len, d_model)
      │
      ▼
╔══════════════════════════════════════╗
║   TransformerEncoder  ×  N=6         ║
║  ┌─────────────────────────────────┐ ║
║  │  ScaledDotProductAttention      │ ║
║  │  softmax( Q·Kᵀ / √dₖ ) · V     │ ║
║  ├─────────────────────────────────┤ ║
║  │  Add & Norm  (residual #1)      │ ║
║  ├─────────────────────────────────┤ ║
║  │  FeedForwardNetwork             │ ║
║  │  max(0, x·W₁+b₁)·W₂ + b₂      │ ║
║  ├─────────────────────────────────┤ ║
║  │  Add & Norm  (residual #2)      │ ║
║  └─────────────────────────────────┘ ║
╚══════════════════════════════════════╝
      │
      ▼
Vetor Z : (batch=1, seq_len, d_model)   ← representação contextualizada
```

---

## Hiperparâmetros

| Parâmetro | Valor (laboratório) | Valor (paper) |
|-----------|--------------------:|------------:|
| `d_model` | 64 | 512 |
| `d_ff` | 256 | 2048 |
| `n_layers` | 6 | 6 |
| `epsilon` | 1e-6 | — |

Todos os hiperparâmetros estão centralizados em `config.py` via `TransformerConfig`.

---

## Pré-requisitos

```
Python >= 3.10
numpy
pandas
pytest   (opcional — para os testes unitários)
```

Instalação das dependências:

```bash
pip install numpy pandas pytest
```

---

## Como Executar

### Pipeline principal

```bash
python main.py
```

Saída esperada:

```
════════════════════════════════════════════════════════════
  TRANSFORMER ENCODER — FROM SCRATCH
  iCEV · Tópicos em Inteligência Artificial 2026.1
════════════════════════════════════════════════════════════

[PASSO 1 · PREPARAÇÃO DOS DADOS]
  Frase de entrada : 'o banco bloqueou meu cartao de credito'
  Tensor X shape   : (1, 7, 64)

[PASSO 2 & 3 · ENCODER STACK (N=6 CAMADAS)]
  [Camada 01/6] shape: (1, 7, 64)
  ...
  [Camada 06/6] shape: (1, 7, 64)

[VALIDAÇÃO DE SANIDADE]
  Shape de entrada (X) : (1, 7, 64)
  Shape de saída   (Z) : (1, 7, 64)  ✓  dimensões preservadas
```

### Testes unitários

```bash
pytest tests/ -v
```

Cobre: `softmax`, `relu`, `layer_norm`, `ScaledDotProductAttention`,
`FeedForwardNetwork`, `EncoderLayer`, `TransformerEncoder` e `DataPipeline`
— totalizando **21 asserções**.

---

## Componentes

### `config.py` — `TransformerConfig`
Dataclass imutável (`frozen=True`) com todos os hiperparâmetros.
Importe `DEFAULT_CONFIG` para usar os valores padrão.

### `data_pipeline.py` — `DataPipeline`
Encapsula vocabulário, tokenização e construção do tensor de entrada.
Suporta vocabulários customizados via parâmetro `vocab`.

### `transformer/utils/` — Funções puras
| Função | Descrição |
|--------|-----------|
| `softmax(x, axis)` | Softmax numericamente estável (subtração do máximo) |
| `relu(x)` | Rectified Linear Unit via `np.maximum(0, x)` |
| `layer_norm(x, epsilon)` | Normalização por token ao longo do eixo de features |

### `transformer/layers/` — Sub-camadas
| Classe | Responsabilidade |
|--------|-----------------|
| `ScaledDotProductAttention` | Projeções W_Q/W_K/W_V · produto escalar · softmax |
| `FeedForwardNetwork` | Duas projeções lineares com ReLU entre elas |
| `EncoderLayer` | Combina Attention + FFN + Add & Norm |

### `transformer/encoder.py` — `TransformerEncoder`
Pilha de `N` `EncoderLayer`s sequenciais com validação de shape.
