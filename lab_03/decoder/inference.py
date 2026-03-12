import numpy as np
from utils.activations import softmax


VOCAB_SIZE: int = 10_000

VOCABULARY: list[str] = (
    ["<PAD>", "<START>", "<EOS>"]
    + [f"palavra_{i}" for i in range(VOCAB_SIZE - 3)]
)

TOKEN_TO_ID: dict[str, int] = {tok: idx for idx, tok in enumerate(VOCABULARY)}
EOS_TOKEN:   str            = "<EOS>"
START_TOKEN: str            = "<START>"
EOS_ID:      int            = TOKEN_TO_ID[EOS_TOKEN]

def generate_next_token(
    current_sequence: list[str],
    encoder_out: np.ndarray,
) -> np.ndarray:
   
    np.random.seed(len(current_sequence) * 7)
    logits = np.random.randn(VOCAB_SIZE)

    if len(current_sequence) >= 6:
        logits[EOS_ID] = 100.0

    return softmax(logits)


def autoregressive_loop(
    encoder_out: np.ndarray,
    max_steps: int = 50,
    verbose: bool = True,
) -> list[str]:

    current_sequence: list[str] = [START_TOKEN]

    if verbose:
        print(f"\n  {'Passo':<8} {'Token Gerado':<20} {'P(token)':>10}")
        print(f"  {'─'*42}")

    for step in range(1, max_steps + 1):
        probs      = generate_next_token(current_sequence, encoder_out)
        next_id    = int(np.argmax(probs))
        next_token = VOCABULARY[next_id]
        prob_value = probs[next_id]

        current_sequence.append(next_token)

        if verbose:
            print(f"  {step:<8} {next_token:<20} {prob_value:>10.6f}")

        if next_token == EOS_TOKEN:
            break

    return current_sequence
