
import numpy as np
import pandas as pd
from config import TransformerConfig, DEFAULT_CONFIG


_DEFAULT_VOCAB: dict[str, int] = {
    "o":        0,
    "banco":    1,
    "bloqueou": 2,
    "cartao":   3,
    "meu":      4,
    "de":       5,
    "credito":  6,
}


class DataPipeline:

    def __init__(
        self,
        vocab:  dict[str, int] | None = None,
        config: TransformerConfig = DEFAULT_CONFIG,
    ) -> None:
        self.config = config
        self._vocab = vocab or _DEFAULT_VOCAB

        self.vocab_df = self._build_vocab_df()

        self._word2id: dict[str, int] = dict(
            zip(self.vocab_df["palavra"], self.vocab_df["id"])
        )

        np.random.seed(config.seed)
        self.embedding_table = np.random.randn(
            len(self._vocab), config.d_model
        )


    def tokenize(self, sentence: str) -> list[int]:
      
        tokens = sentence.lower().split()
        return [self._word2id[t] for t in tokens if t in self._word2id]

    def build_input_tensor(self, ids: list[int]) -> np.ndarray:
       
        if not ids:
            raise ValueError("Lista de IDs vazia. Verifique a frase de entrada.")

        embeddings = self.embedding_table[ids]     
        return embeddings[np.newaxis, ...]      

    def summary(self) -> None:
        print("┌─ Vocabulário ─────────────────────────────┐")
        print(self.vocab_df.to_string(index=False))
        print(f"└─ vocab_size={len(self._vocab)} │ "
              f"embedding_table={self.embedding_table.shape} ─┘")

    def _build_vocab_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            list(self._vocab.items()),
            columns=["palavra", "id"],
        )
