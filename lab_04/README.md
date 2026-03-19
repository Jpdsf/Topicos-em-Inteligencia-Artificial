# Lab 04 — O Transformer Completo "From Scratch"

**Disciplina:** Tópicos em Inteligência Artificial — iCEV 2026.1  
**Professor:** Dimmy Magalhães  

> **Nota de IA:** Partes geradas/complementadas com IA, revisadas por João Paulo.

---

## Visão Geral

Implementação completa da arquitetura **Encoder-Decoder Transformer** em NumPy puro,
conforme Vaswani et al. (2017) — *"Attention Is All You Need"*.

O projeto integra os módulos construídos nos Labs 01–03 em uma topologia coerente,
culminando no laço auto-regressivo de inferência (Tarefa 4).

---

## Estrutura do Projeto

```
transformer_lab04/
├── transformer/                  # Pacote principal
│   ├── __init__.py
│   ├── attention/
│   │   ├── scaled_dot_product.py # Tarefa 1.1 — Scaled Dot-Product Attention
│   │   └── multihead.py          # Multi-Head Attention
│   ├── utils/
│   │   ├── ffn.py                # Tarefa 1.2 — Position-wise FFN
│   │   ├── add_norm.py           # Tarefa 1.3 — Add & Norm (residual + LayerNorm)
│   │   ├── positional_encoding.py
│   │   └── mask.py               # Máscara causal (Lab 02)
│   ├── encoder/
│   │   ├── encoder_block.py      # Tarefa 2 — EncoderBlock
│   │   └── encoder.py            # Pilha de N EncoderBlocks
│   └── decoder/
│       ├── decoder_block.py      # Tarefa 3 — DecoderBlock
│       └── decoder.py            # Pilha + projeção Linear + Softmax
├── scripts/
│   └── inference.py              # Tarefa 4 — laço auto-regressivo
├── tests/
│   └── test_transformer.py       # Testes unitários de todos os componentes
└── README.md
```

---

## Lógica Matemática dos Componentes

### 1. Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```
- A divisão por `√d_k` evita que o produto interno cresça com a dimensionalidade,
  estabilizando os gradientes do softmax.
- A máscara é **aditiva** (−∞ antes do softmax), zerando atenção sobre posições futuras.

### 2. Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · W_O
head_i = Attention(Q·W_Qᵢ, K·W_Kᵢ, V·W_Vᵢ)
```
- Divide `d_model` em `h` subespaços de `d_k = d_model / h` cada.
- Permite que o modelo capture diferentes tipos de relações em paralelo.

### 3. Position-wise FFN
```
FFN(x) = ReLU(x·W₁ + b₁)·W₂ + b₂
```
- Expansão: `d_model → d_ff → d_model` (tipicamente `d_ff = 4 × d_model`).
- Aplicada **independentemente** em cada posição da sequência.

### 4. Add & Norm
```
Output = LayerNorm(x + Sublayer(x))
```
- A **conexão residual** permite gradientes fluírem diretamente pelas camadas profundas.
- A **LayerNorm** normaliza ao longo do eixo `d_model` (não do batch), estabilizando
  a distribuição de ativações durante o treinamento.

### 5. Máscara Causal
- Matriz triangular superior com `−∞` nas posições acima da diagonal.
- Impede que o Decoder veja tokens futuros: a previsão em `t` depende apenas de `t−1, t−2, ...`.

### 6. Fluxo do EncoderBlock
```
x → MultiHeadSelfAttention(x, x, x) → Add&Norm → FFN → Add&Norm → Z
```
- Sem máscara: atenção **bidirecional** — cada token vê toda a sequência.

### 7. Fluxo do DecoderBlock
```
y → MaskedSelfAttention(y, y, y, causal_mask) → Add&Norm
  → CrossAttention(Q=y, K=Z, V=Z)             → Add&Norm
  → FFN                                        → Add&Norm
```
- **Masked Self-Attention:** o Decoder não trapaceia durante treinamento.
- **Cross-Attention:** conecta cada posição do Decoder à memória completa do Encoder.

### 8. Laço Auto-regressivo
```
decoder_ids = [<START>]
while True:
    probs = Decoder(decoder_ids, Z)
    next_token = argmax(probs[:, -1, :])
    decoder_ids.append(next_token)
    if next_token == <EOS>: break
```
- A cada passo, o modelo consome **todos** os tokens gerados até então.
- O Causal Mask garante que, durante esse processo, cada posição só atenda ao passado.

---

## Como Executar

### Pré-requisitos
```bash
pip install numpy pytest
```

### Inferência (Tarefa 4)
```bash
python scripts/inference.py
```

### Testes unitários
```bash
python -m pytest tests/ -v
```

---

## Dependências

| Pacote  | Versão mínima | Uso                        |
|---------|---------------|----------------------------|
| numpy   | 1.24          | Toda a álgebra tensorial   |
| pytest  | 7.0           | Testes unitários (opcional)|

---

## Uso de IA

Este projeto utilizou IA para brainstorming e geração de templates básicos (estrutura de arquivos, esqueleto de funções/classes). A lógica matemática dos componentes (fórmulas, fluxos Encoder/Decoder, máscara causal) foi documentada e implementada pelo autor, conforme a seção "Lógica Matemática dos Componentes" acima.

---

## Referência

Vaswani, A. et al. (2017). **Attention Is All You Need**.  
*Advances in Neural Information Processing Systems*, 30.  
https://arxiv.org/abs/1706.03762
