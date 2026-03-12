import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from decoder.inference import autoregressive_loop


def run():
    print("TAREFA 3 — Loop de Inferência Auto-Regressivo")

    np.random.seed(1)
    encoder_out = np.random.randn(1, 10, 512)

    print("\nIniciando geração token a token...\n")
    sequence = autoregressive_loop(encoder_out, max_steps=20, verbose=True)

    frase_final = " ".join(sequence)
    print(f"\n Token <EOS> detectado — geração encerrada.")
    print(f"\n Frase Final Gerada:")
    print(f"   {frase_final}")


if __name__ == "__main__":
    run()
