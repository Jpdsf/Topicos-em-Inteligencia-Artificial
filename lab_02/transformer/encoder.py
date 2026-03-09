import numpy as np
from config import TransformerConfig, DEFAULT_CONFIG
from transformer.layers.encoder_layer import EncoderLayer


class TransformerEncoder:
  
    def __init__(self, config: TransformerConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        np.random.seed(config.seed)

        self.layers: list[EncoderLayer] = [
            EncoderLayer(
                d_model = config.d_model,
                d_ff    = config.d_ff,
                epsilon = config.epsilon,
            )
            for _ in range(config.n_layers)
        ]

    def forward(self, X: np.ndarray, verbose: bool = False) -> np.ndarray:
       
        self._validate_input(X)

        for i, layer in enumerate(self.layers):
            X = layer.forward(X)
            if verbose:
                print(f"  [Camada {i + 1:02d}/{self.config.n_layers}] "
                      f"shape: {X.shape}")
        return X

    def _validate_input(self, X: np.ndarray) -> None:
        if X.ndim != 3:
            raise ValueError(
                f"Tensor de entrada deve ter 3 dimensões "
                f"(batch, seq_len, d_model), recebeu: {X.ndim}D."
            )
        if X.shape[-1] != self.config.d_model:
            raise ValueError(
                f"Última dimensão do tensor ({X.shape[-1]}) "
                f"não corresponde a d_model ({self.config.d_model})."
            )
