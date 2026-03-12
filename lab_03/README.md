# Laboratório 3 — Implementando o Decoder
**Instituto de Ensino Superior ICEV**

Implementação modular dos blocos matemáticos centrais do Decoder Transformer.

---

## Estrutura do Projeto

```
lab3_decoder/
│
├── main.py                      # Ponto de entrada: roda demos + testes
│
├── decoder/                     # Módulos principais
│   ├── mask.py                  # Tarefa 1 — Máscara Causal (Look-Ahead Mask)
│   ├── attention.py             # Tarefa 2 — Scaled Dot-Product + Cross-Attention
│   └── inference.py             # Tarefa 3 — Loop de Inferência Auto-Regressivo
│
├── utils/                       # Utilitários reutilizáveis
│   ├── activations.py           # Softmax numericamente estável
│   └── projections.py           # Projeção linear
│
├── demos/                       # Scripts de demonstração por tarefa
│   ├── demo_mask.py
│   ├── demo_cross_attention.py
│   └── demo_inference.py
│
└── tests/                       # Testes unitários
    ├── test_mask.py
    ├── test_attention.py
    └── test_inference.py
```

---

## Como Executar

### Tudo de uma vez (demos + testes)
```bash
python main.py
```

### Apenas uma tarefa
```bash
python demos/demo_mask.py
python demos/demo_cross_attention.py
python demos/demo_inference.py
```

### Apenas os testes
```bash
python tests/test_mask.py
python tests/test_attention.py
python tests/test_inference.py
```

---

## Tarefas Implementadas

### Tarefa 1 — Máscara Causal (`decoder/mask.py`)
Cria a matriz `M` que impede o modelo de "olhar para o futuro":
- Triangular inferior + diagonal → `0`
- Triangular superior → `-inf`

Após o Softmax, os `-inf` se tornam exatamente `0.0`.

### Tarefa 2 — Cross-Attention (`decoder/attention.py`)
Ponte entre Encoder e Decoder:
- **Query (Q)** vem do estado atual do Decoder
- **Key (K) e Value (V)** vêm da saída do Encoder
- Sem máscara causal — acesso total à frase original

### Tarefa 3 — Inferência Auto-Regressiva (`decoder/inference.py`)
Loop `while` que gera tokens um por vez:
1. Chama o mock do Decoder para obter `P(próximo token)`
2. `argmax` seleciona o token mais provável
3. Appenda ao contexto e repete
4. Para ao encontrar `<EOS>`

---

## Dependências

```
numpy
```

Instalar: `pip install numpy`
