# Tarefa 2: O Loop de Fusão
#
# Após identificar o par mais frequente, o algoritmo funde os dois símbolos
# em um único token novo e atualiza o vocabulário com essa mudança.

import re


def merge_vocab(pair, v_in):
    # Substitui todas as ocorrências isoladas do par mais frequente pela
    # versão unificada, retornando o vocabulário atualizado
    v_out = {}
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = pattern.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def run_bpe(vocab, num_merges=5):
    # Executa o loop principal do BPE: a cada iteração encontra o par mais
    # frequente, realiza a fusão e imprime o estado atual do vocabulário
    from tarefa1 import get_stats

    for i in range(num_merges):
        stats = get_stats(vocab)
        best_pair = max(stats, key=stats.get)
        vocab = merge_vocab(best_pair, vocab)
        print(f"Iteração {i + 1}: par fundido -> {best_pair}")
        print(f"Vocabulário atual: {vocab}")
        print()

    return vocab
