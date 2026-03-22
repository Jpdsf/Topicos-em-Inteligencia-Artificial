from lab_05.data import load_translation_subset
from lab_05.tokenization import Tokenizer
from lab_05.training import Trainer
from lab_05.inference import overfit_test
from lab_05.config import DATA_NUM_SAMPLES


def main():
    # Tarefa 1: carrega dataset
    print("=" * 60)
    print("TAREFA 1 — Dataset")
    print("=" * 60)
    pairs = load_translation_subset(DATA_NUM_SAMPLES)

    # Tarefa 2: tokeniza
    print("\n" + "=" * 60)
    print("TAREFA 2 — Tokenização")
    print("=" * 60)
    tok = Tokenizer()
    src_ids, tgt_input_ids, tgt_target_ids = tok.encode_batch(pairs)
    print(f"  src shape    : {len(src_ids)} x {len(src_ids[0])}")
    print(f"  tgt_in shape : {len(tgt_input_ids)} x {len(tgt_input_ids[0])}")
    print(f"  tgt_out shape: {len(tgt_target_ids)} x {len(tgt_target_ids[0])}")

    # Tarefa 3: training loop
    print("\n" + "=" * 60)
    print("TAREFA 3 — Training Loop")
    print("=" * 60)
    trainer = Trainer()
    history = trainer.train(src_ids, tgt_input_ids, tgt_target_ids)
    print(f"\n  Loss inicial : {history[0]:.4f}")
    print(f"  Loss final   : {history[-1]:.4f}")
    print(
        f"  Redução      : {((history[0] - history[-1]) / history[0]) * 100:.1f}%")

    # Tarefa 4: overfitting test
    print("\n" + "=" * 60)
    print("TAREFA 4 — Overfitting Test")
    print("=" * 60)
    src_sentence = pairs[0]["src"]
    tgt_sentence = pairs[0]["tgt"]
    overfit_test(src_sentence, tgt_sentence)


if __name__ == "__main__":
    main()
