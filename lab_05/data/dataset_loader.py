from datasets import load_dataset
from lab_05.config import (
    DATA_DATASET_NAME,
    DATA_NUM_SAMPLES,
    DATA_SPLIT,
    DATA_SRC_LANG,
    DATA_TGT_LANG,
)
from dotenv import load_dotenv

load_dotenv()


def load_translation_subset(
    num_samples: int = DATA_NUM_SAMPLES,
) -> list[dict]:
    dataset = load_dataset(DATA_DATASET_NAME, split=DATA_SPLIT)
    subset = dataset.select(range(num_samples))

    pairs = [
        {
            "src": example[DATA_SRC_LANG],
            "tgt": example[DATA_TGT_LANG],
        }
        for example in subset
    ]

    print(f"[Dataset] {len(pairs)} pares carregados.")
    print(f"  src[0]: {pairs[0]['src']}")
    print(f"  tgt[0]: {pairs[0]['tgt']}")

    return pairs
