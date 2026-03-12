import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from decoder.attention import cross_attention


def run():
    print("TAREFA 2 — Cross-Attention (Ponte Encoder-Decoder)")

    batch_size = 1
    seq_enc    = 10   # tokens da frase em francês (Encoder)
    seq_dec    = 4    # tokens já gerados em inglês (Decoder)
    d_model    = 512
    d_k        = 64
    d_v        = 64

    np.random.seed(0)
    encoder_out   = np.random.randn(batch_size, seq_enc, d_model)
    decoder_state = np.random.randn(batch_size, seq_dec, d_model)

    W_q = np.random.randn(d_model, d_k) * 0.01
    W_k = np.random.randn(d_model, d_k) * 0.01
    W_v = np.random.randn(d_model, d_v) * 0.01

    print(f"\nEncoder output shape : {encoder_out.shape}   (francês)")
    print(f"Decoder state  shape : {decoder_state.shape}    (inglês gerado)")

    output = cross_attention(encoder_out, decoder_state, W_q, W_k, W_v)

    print(f"\nSaída do Cross-Attention : {output.shape}")
    print(f"  → [batch={batch_size}, seq_dec={seq_dec}, d_v={d_v}]")
    print(f"\nPrimeiros valores (batch=0, token=0, dims 0-7):")
    print(np.round(output[0, 0, :8], 6), "...")
    print("\n Cross-Attention calculado — sem máscara causal (acesso total ao Encoder)")


if __name__ == "__main__":
    run()
