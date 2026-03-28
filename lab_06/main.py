# Laboratório 6 - P2: Construindo um Tokenizador BPE e Explorando o WordPiece
# Aluno: João Paulo
# iCEV

from tarefa1 import vocab, get_stats
from tarefa2 import run_bpe
from tarefa3 import run_wordpiece


def main():
    print("=" * 55)
    print("Tarefa 1: Motor de Frequências")
    print("=" * 55)

    stats = get_stats(vocab)
    print(f"Frequência do par ('e', 's'): {stats[('e', 's')]}")  # deve ser 9
    print()

    print("=" * 55)
    print("Tarefa 2: Loop de Fusão (5 iterações)")
    print("=" * 55)

    run_bpe(vocab, num_merges=5)

    print("=" * 55)
    print("Tarefa 3: WordPiece com BERT multilíngue")
    print("=" * 55)

    run_wordpiece()


if __name__ == "__main__":
    main()