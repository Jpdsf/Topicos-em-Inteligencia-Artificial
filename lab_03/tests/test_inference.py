import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from decoder.inference import (
    generate_next_token,
    autoregressive_loop,
    VOCAB_SIZE,
    START_TOKEN,
    EOS_TOKEN,
)


def test_next_token_prob_shape():
    enc = np.random.randn(1, 10, 512)
    probs = generate_next_token([START_TOKEN], enc)
    assert probs.shape == (VOCAB_SIZE,), f"Shape incorreto: {probs.shape}"
    print("  [OK] test_next_token_prob_shape")


def test_next_token_probs_sum_to_one():
    enc = np.random.randn(1, 10, 512)
    probs = generate_next_token([START_TOKEN, "palavra_1"], enc)
    assert np.isclose(probs.sum(), 1.0, atol=1e-6), f"Soma != 1: {probs.sum()}"
    print("  [OK] test_next_token_probs_sum_to_one")


def test_loop_starts_with_start_token():
    enc = np.random.randn(1, 10, 512)
    seq = autoregressive_loop(enc, max_steps=20, verbose=False)
    assert seq[0] == START_TOKEN, f"Primeiro token incorreto: {seq[0]}"
    print("  [OK] test_loop_starts_with_start_token")


def test_loop_ends_with_eos():
    np.random.seed(1)
    enc = np.random.randn(1, 10, 512)
    seq = autoregressive_loop(enc, max_steps=50, verbose=False)
    assert seq[-1] == EOS_TOKEN, f"Último token não é <EOS>: {seq[-1]}"
    print("  [OK] test_loop_ends_with_eos")


def test_loop_respects_max_steps():
    enc = np.random.randn(1, 10, 512)
    np.random.seed(999)
    seq = autoregressive_loop(enc, max_steps=2, verbose=False)
    assert len(seq) <= 3, f"Loop não respeitou max_steps: len={len(seq)}"
    print("  [OK] test_loop_respects_max_steps")


def run_all():
    print("\n── Testes: Inferência Auto-Regressiva ──")
    test_next_token_prob_shape()
    test_next_token_probs_sum_to_one()
    test_loop_starts_with_start_token()
    test_loop_ends_with_eos()
    test_loop_respects_max_steps()
    print("  Todos os testes passaram!\n")


if __name__ == "__main__":
    run_all()
