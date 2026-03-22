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

        src = self._to_tensor(src_ids)
        tgt_input = self._to_tensor(tgt_input_ids)
        tgt_target = self._to_tensor(tgt_target_ids)

        loss_history = []

        self.model.train()
        for epoch in range(1, TRAIN_EPOCHS + 1):
            self.optimizer.zero_grad()

            output = self.model(src, tgt_input)

            batch_size, seq_len, vocab_size = output.shape
            output_flat = output.reshape(batch_size * seq_len, vocab_size)
            target_flat = tgt_target.reshape(batch_size * seq_len)

            loss = self.criterion(output_flat, target_flat)

            loss.backward()
            self.optimizer.step()

            loss_val = loss.item()
            loss_history.append(loss_val)
            print(f"  Epoch {epoch:>3}/{TRAIN_EPOCHS} | Loss: {loss_val:.4f}")

        return loss_history
