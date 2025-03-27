# Tarefa 1: O Motor de Frequências
#
# O corpus de treinamento representa palavras já separadas em caracteres.
# O símbolo </w> marca o fim de cada palavra.

vocab = {
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3
}


def get_stats(vocab):
    # Percorre cada palavra do vocabulário e conta a frequência de todos
    # os pares de símbolos adjacentes, ponderando pela frequência da palavra
    pairs = {}
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs
