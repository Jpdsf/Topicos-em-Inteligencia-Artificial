"""
main.py
--------
Ponto de entrada do laboratório.

Orquestra o pipeline completo:
    Dados → Tokenização → Embedding → Encoder Stack → Vetor Z

Uso
---
    python main.py
"""

import numpy as np

from config import TransformerConfig
from data_pipeline import DataPipeline
from transformer.encoder import TransformerEncoder


# ─── Separador visual ─────────────────────────────────────────────────────────
_LINE  = "═" * 60
_DLINE = "─" * 60


def print_header() -> None:
    print(_LINE)
    print("  TRANSFORMER ENCODER — FROM SCRATCH")
    print("  iCEV · Tópicos em Inteligência Artificial 2026.1")
    print("  Prof. Dimmy Magalhães")
    print(_LINE)


def print_section(title: str) -> None:
    print(f"\n[{title}]")
    print(_DLINE)


def run() -> np.ndarray:
    """
    Executa o pipeline completo e retorna o vetor Z final.

    Retorna
    -------
    Z : np.ndarray  (batch=1, seq_len, d_model)
    """
    print_header()

    # ── Configuração ──────────────────────────────────────────────
    print_section("CONFIGURAÇÃO")
    config = TransformerConfig(
        d_model  = 64,
        d_ff     = 256,
        n_layers = 6,
        epsilon  = 1e-6,
        seed     = 42,
    )
    print(f"  d_model  : {config.d_model}  (paper: 512)")
    print(f"  d_ff     : {config.d_ff}  (paper: 2048)")
    print(f"  n_layers : {config.n_layers}")
    print(f"  epsilon  : {config.epsilon}")
    print(f"  seed     : {config.seed}")

    # ── Passo 1: Dados ────────────────────────────────────────────
    print_section("PASSO 1 · PREPARAÇÃO DOS DADOS")

    pipeline = DataPipeline(config=config)
    pipeline.summary()

    frase = "o banco bloqueou meu cartao de credito"
    ids   = pipeline.tokenize(frase)
    X     = pipeline.build_input_tensor(ids)

    print(f"\n  Frase de entrada : '{frase}'")
    print(f"  Token IDs        : {ids}")
    print(f"  Tensor X shape   : {X.shape}  "
          f"→ (batch=1, seq_len={len(ids)}, d_model={config.d_model})")

    # ── Passo 2 & 3: Encoder ──────────────────────────────────────
    print_section("PASSO 2 & 3 · ENCODER STACK (N=6 CAMADAS)")

    encoder = TransformerEncoder(config)
    Z = encoder.forward(X, verbose=True)

    # ── Validação de sanidade ─────────────────────────────────────
    print_section("VALIDAÇÃO DE SANIDADE")

    assert Z.shape == X.shape, (
        f"FALHA: shape de entrada {X.shape} ≠ shape de saída {Z.shape}"
    )
    print(f"  Shape de entrada (X) : {X.shape}")
    print(f"  Shape de saída   (Z) : {Z.shape}  ✓  dimensões preservadas")
    print(f"\n  Norma L2 média por token   : "
          f"{np.linalg.norm(Z, axis=-1).mean():.4f}")
    print(f"  Primeiros 8 valores de Z[0,0] : "
          f"{Z[0, 0, :8].round(4)}")

    print(f"\n{'═' * 60}")
    print("  Pipeline concluído com sucesso.")
    print("═" * 60)

    return Z


if __name__ == "__main__":
    run()
