import torch
import torch.nn as nn
from torch.optim import Adam

from lab_05.transformer import Transformer
from lab_05.config import (
    MODEL_D_MODEL,
    MODEL_NUM_HEADS,
    MODEL_NUM_LAYERS,
    MODEL_D_FF,
    MODEL_DROPOUT,
    MODEL_VOCAB_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TOKENIZER_PAD_ID,
    TRAIN_BATCH_SIZE
)


class Trainer:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = Transformer(
            vocab_size=MODEL_VOCAB_SIZE,
            d_model=MODEL_D_MODEL,
            num_heads=MODEL_NUM_HEADS,
            num_layers=MODEL_NUM_LAYERS,
            d_ff=MODEL_D_FF,
            dropout=MODEL_DROPOUT,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=TOKENIZER_PAD_ID)
        self.optimizer = Adam(self.model.parameters(), lr=TRAIN_LR)

    def _to_tensor(self, sequences: list[list[int]]) -> torch.Tensor:
        return torch.tensor(sequences, dtype=torch.long).to(self.device)

    def train(
        self,
        src_ids: list[list[int]],
        tgt_input_ids: list[list[int]],
        tgt_target_ids: list[list[int]],
    ) -> list[float]:
        loss_history = []

        self.model.train()
        for epoch in range(1, TRAIN_EPOCHS + 1):
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(src_ids), TRAIN_BATCH_SIZE):
                src = self._to_tensor(src_ids[i:i + TRAIN_BATCH_SIZE])
                tgt_input = self._to_tensor(
                    tgt_input_ids[i:i + TRAIN_BATCH_SIZE])
                tgt_target = self._to_tensor(
                    tgt_target_ids[i:i + TRAIN_BATCH_SIZE])

                self.optimizer.zero_grad()
                output = self.model(src, tgt_input)

                batch_size, seq_len, vocab_size = output.shape
                loss = self.criterion(
                    output.reshape(batch_size * seq_len, vocab_size),
                    tgt_target.reshape(batch_size * seq_len),
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)
            print(f"  Epoch {epoch:>3}/{TRAIN_EPOCHS} | Loss: {avg_loss:.4f}")

        return loss_history
