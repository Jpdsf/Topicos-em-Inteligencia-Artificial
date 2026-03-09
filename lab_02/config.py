from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    
    d_model:  int   = 64
    d_ff:     int   = 256
    n_layers: int   = 6
    epsilon:  float = 1e-6
    seed:     int   = 42


DEFAULT_CONFIG = TransformerConfig()
