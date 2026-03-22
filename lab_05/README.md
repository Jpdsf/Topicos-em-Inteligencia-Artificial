# Lab 05 — Treinamento Fim-a-Fim do Transformer

Laboratório final da Unidade I. Integra dataset real, tokenização,
loop de treinamento e inferência autoregressiva em um pipeline completo.

## Estrutura
```
lab_05/
├── config.py              # hiperparâmetros centralizados
├── main.py                # entrypoint geral
├── data/                  # Tarefa 1: carregamento do dataset
├── tokenization/          # Tarefa 2: tokenização e padding
├── transformer/           # Transformer PyTorch (encoder + decoder)
├── training/              # Tarefa 3: training loop
├── inference/             # Tarefa 4: overfitting test
└── tests/                 # testes por tarefa
```

## Como executar
```bash
# pipeline completo
python -m lab_05.main

# testes
python -m pytest lab_05/tests/ -v
```

## Tarefas

### Tarefa 1 — Dataset
Carrega o dataset `bentrevett/multi30k` do Hugging Face e seleciona
1.000 pares inglês→alemão para treinamento.

### Tarefa 2 — Tokenização
Usa `bert-base-multilingual-cased` para converter texto em IDs.
Adiciona tokens especiais `[CLS]` como `<START>` e `[SEP]` como `<EOS>`
nas sequências do decoder. Aplica padding para uniformizar o batch.

### Tarefa 3 — Training Loop
Instancia o Transformer com `d_model=128`, `4 heads`, `2 layers`.
Loop de 15 épocas com `CrossEntropyLoss` (ignore_index=0 para padding)
e otimizador `Adam`. A curva de loss cai de 8.94 para 1.73 (redução de 80.6%).

### Tarefa 4 — Overfitting Test
Treina o modelo em um único par por 300 épocas e gera a tradução
autoregressivamente via greedy decoding. O modelo reproduz a frase
alvo com fidelidade, provando que os gradientes fluem corretamente.

## Decisão Arquitetural

O Lab 04 solicitava que as classes fossem reescritas "garantindo que
aceitem tensores dinâmicos de PyTorch ou NumPy". As implementações
dos Labs 02–04 foram entregues em NumPy puro, sem suporte a gradientes.

Para cumprir a Tarefa 3 do Lab 05 (backpropagation com Adam), as
classes foram portadas para PyTorch `nn.Module` em `lab_05/transformer/`,
mantendo a mesma arquitetura e lógica matemática dos labs anteriores:

| Lab 04 (NumPy)        | Lab 05 (PyTorch)        | Lógica preservada              |
|-----------------------|-------------------------|--------------------------------|
| `MultiHeadAttention`  | `nn.MultiheadAttention` | Q, K, V, W_O, split/merge      |
| `EncoderBlock`        | `EncoderBlock`          | self-attn → Add&Norm → FFN     |
| `Encoder`             | `Encoder`               | N blocos empilhados            |
| `DecoderBlock`        | `DecoderBlock`          | masked → cross-attn → FFN      |
| `Decoder`             | `Decoder`               | N blocos + projeção final      |
| `PositionWiseFFN`     | `nn.Sequential`         | Linear → ReLU → Linear         |
| `add_and_norm`        | `nn.LayerNorm`          | x + sublayer(x), normalizado   |
| `positional_encoding` | `PositionalEncoding`    | sin/cos idênticos ao paper     |
| `make_causal_mask`    | `_causal_mask`          | triangular superior            |

## Ferramentas de IA utilizadas

- **Claude (Anthropic)** — estruturação do pipeline, tokenização
  (Tarefas 1 e 2), debugging e port das classes NumPy → PyTorch.
- Partes geradas/complementadas com IA, revisadas por João Paulo.
- O fluxo Forward/Backward (Tarefa 3) replica fielmente a arquitetura
  construída nos Labs 02, 03 e 04, agora com suporte a autograd.

## Variáveis de Ambiente

Este projeto utiliza a API do Hugging Face para baixar o dataset e o tokenizador.
Sem autenticação, você verá um aviso de rate limit. Para resolvê-lo, crie um
arquivo `.env` na raiz do projeto com sua chave do Hugging Face:
```
HF_TOKEN=sua_chave_aqui
```

Você pode obter uma chave gratuita em [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).