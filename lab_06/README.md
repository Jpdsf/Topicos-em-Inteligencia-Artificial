# Laboratório 6 - P2: Construindo um Tokenizador BPE e Explorando o WordPiece

> **Instituto de Ensino Superior ICEV** — Disciplina de Inteligência Artificial

---

## Estrutura do Projeto

```
lab6-tokenizer/
│
├── main.py          # Ponto de entrada: executa todas as tarefas
│
├── tarefa1.py       # Tarefa 1: vocab + get_stats()
│
├── tarefa2.py       # Tarefa 2: merge_vocab() + run_bpe() (K=5)
│
├── tarefa3.py       # Tarefa 3: WordPiece com BERT multilíngue
│
├── requirements.txt
│
└── README.md
```

---

## Como executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Executar todas as tarefas

```bash
python main.py
```

### 3. Executar tarefas individualmente

```bash
# Tarefa 1 — Motor de Frequências
python tarefa1.py

# Tarefa 2 — Loop de Fusão BPE
python tarefa2.py

# Tarefa 3 — WordPiece BERT
python tarefa3.py
```

---

## Resumo de cada Tarefa

### Tarefa 1 — Motor de Frequências

A função `get_stats(vocab)` percorre o corpus BPE e conta a frequência de todos os pares adjacentes de símbolos, multiplicando pelo número de ocorrências de cada palavra. O par `('e', 's')` retorna 9 (6 em *newest* + 3 em *widest*), validando a implementação.

### Tarefa 2 — Loop de Fusão (K = 5 iterações)

A função `merge_vocab(pair, v_in)` usa uma expressão regular para localizar o par mais frequente **isolado** no vocabulário e substituí-lo pelo token fundido. O loop principal `run_bpe` executa 5 rodadas. Após as iterações, o sufixo morfológico `est</w>` emerge naturalmente, demonstrando como o BPE descobre estrutura morfológica sem supervisão.

**Sequência de fusões observada:**

| Rodada | Par fundido          | Freq |
|--------|----------------------|------|
| 1      | `('e', 's')`         | 9    |
| 2      | `('es', 't')`        | 9    |
| 3      | `('est', '</w>')`    | 9    |
| 4      | `('l', 'o')`         | 7    |
| 5      | `('lo', 'w')`        | 7    |

### Tarefa 3 — WordPiece com BERT Multilíngue

O tokenizador `bert-base-multilingual-cased` foi carregado via `AutoTokenizer` e utilizado para segmentar a seguinte frase de teste:

> *"Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."*

---

## O que significa o prefixo `##` nos tokens do WordPiece?

Quando o BERT tokeniza uma palavra, ele tenta encontrar a sequência de sub-palavras conhecidas que melhor a representa. O prefixo `##` sinaliza que aquele fragmento **não é o início de uma nova palavra**, mas sim uma **continuação (sufixo)** que se une ao token anterior.

**Exemplo observado na frase de teste:**

A palavra *"inconstitucionalmente"* é decomposta em vários fragmentos, sendo que a maioria carrega o prefixo `##`, por exemplo:

```
'in', '##cons', '##tit', '##uc', '##ional', '##mente'
```

Isso demonstra como palavras longas e raras são fragmentadas em pedaços já conhecidos pelo vocabulário do modelo.

---

## Por que sub-palavras impedem o travamento do modelo?

Modelos baseados em vocabulário de palavras inteiras sofrem do problema **OOV (Out-Of-Vocabulary)**: qualquer palavra não vista durante o treinamento é mapeada para o token especial `[UNK]`, fazendo o modelo perder toda a informação semântica e morfológica daquela posição.

O WordPiece resolve isso com três propriedades:

1. **Cobertura garantida**: qualquer palavra pode ser decomposta em caracteres individuais como último recurso — o modelo *nunca* gera `[UNK]` para texto normal.

2. **Compartilhamento morfológico**: palavras relacionadas como `ajustar`, `ajustável` e `ajuste` compartilham sub-palavras, permitindo que o modelo generalize padrões morfológicos sem ter visto todas as flexões no treinamento.

3. **Eficiência de vocabulário**: ao invés de guardar milhões de formas flexionadas, o modelo trabalha com um conjunto reduzido de sub-palavras que combinadas cobrem praticamente qualquer texto.

---

## Citação de uso de IA Generativa

Conforme exigido pelas instruções do laboratório, declaro que os seguintes trechos foram **gerados com auxílio de IA generativa (Claude — Anthropic)** e posteriormente **revisados e validados manualmente**:

| Arquivo | Trecho | Ferramenta |
|---------|--------|------------|
| `tarefa2.py` | Expressão regular na função `merge_vocab` (padrão com lookahead/lookbehind para garantir que apenas pares isolados sejam substituídos) | Claude (Anthropic) |
| `tarefa2.py` | Estrutura geral da função `merge_vocab` e do loop `run_bpe` | Claude (Anthropic) |

Todos os trechos gerados foram lidos, compreendidos e testados antes da entrega, em conformidade com a política de integridade acadêmica da instituição. Tive o auxílio da IA também para facilitar a elaboração deste documento, mas tudo foi revisado e validado por mim, garantindo que esteja de acordo e alinhado com meu conhecimento sobre a construção de um tokenizador BPE e a exploração do WordPiece.

---

## Referências

- Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units*. ACL 2016.
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL 2019.
- Hugging Face. *Tokenizers documentation*. https://huggingface.co/docs/transformers/tokenizer_summary

---

*Versão: v1.0 — tag obrigatória de entrega*8