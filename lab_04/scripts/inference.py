import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from transformer import Encoder, Decoder, positional_encoding

D_MODEL    = 64
NUM_HEADS  = 4
D_FF       = 256
NUM_LAYERS = 2
BATCH      = 1

VOCAB = {
    "<PAD>":   0,
    "<START>": 1,
    "<EOS>":   2,
    "Thinking":3,
    "Machines":4,
    "Máquinas":5,
    "Pensantes":6,
    "são":     7,
    "incríveis":8,
}
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE  = len(VOCAB)
START_ID    = VOCAB["<START>"]
EOS_ID      = VOCAB["<EOS>"]
MAX_DECODE  = 10


def token_to_embedding(token_ids: list[int], d_model: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    E   = rng.normal(0, 0.1, (VOCAB_SIZE, d_model))

    ids   = np.array(token_ids)
    embed = E[ids]
    return embed[np.newaxis, :, :]


def run_inference() -> None:
    print("=" * 60)
    print(" Transformer Encoder-Decoder — Inferência Toy")
    print("=" * 60)

    encoder = Encoder(D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS)
    decoder = Decoder(D_MODEL, NUM_HEADS, D_FF, VOCAB_SIZE, NUM_LAYERS)

    src_tokens  = [VOCAB["Thinking"], VOCAB["Machines"]]
    src_seq_len = len(src_tokens)

    src_embed   = token_to_embedding(src_tokens, D_MODEL)
    src_pe      = positional_encoding(src_seq_len, D_MODEL)
    encoder_input = src_embed + src_pe

    print(f"\n[Encoder] Entrada: {[ID_TO_TOKEN[t] for t in src_tokens]}")
    Z = encoder(encoder_input)
    print(f"[Encoder] Memória Z gerada — shape: {Z.shape}")

    decoded_ids: list[int] = [START_ID]
    print(f"\n[Decoder] Início com token: <START>")
    print("-" * 40)

    for step in range(MAX_DECODE):
        tgt_seq_len = len(decoded_ids)

        tgt_embed   = token_to_embedding(decoded_ids, D_MODEL)
        tgt_pe      = positional_encoding(tgt_seq_len, D_MODEL)
        decoder_input = tgt_embed + tgt_pe

        probs = decoder(decoder_input, Z)

        next_token_id   = int(np.argmax(probs[0, -1, :]))
        next_token_name = ID_TO_TOKEN.get(next_token_id, f"<ID:{next_token_id}>")

        print(f"  Passo {step + 1:02d} → token previsto: '{next_token_name}' "
              f"(id={next_token_id}, prob={probs[0, -1, next_token_id]:.4f})")

        decoded_ids.append(next_token_id)

        if next_token_id == EOS_ID:
            print("\n[Decoder] Token <EOS> gerado. Inferência encerrada.")
            break
    else:
        print(f"\n[Decoder] Limite de {MAX_DECODE} passos atingido.")

    output_tokens = [ID_TO_TOKEN.get(i, f"<ID:{i}>") for i in decoded_ids]
    print("\n" + "=" * 60)
    print(f" Sequência gerada: {' '.join(output_tokens)}")
    print("=" * 60)


if __name__ == "__main__":
    run_inference()
