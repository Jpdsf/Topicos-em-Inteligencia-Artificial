import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .utils import PositionalEncoding, make_pad_mask
from lab_05.config import TOKENIZER_PAD_ID


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model

        self.src_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=TOKENIZER_PAD_ID)
        self.tgt_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=TOKENIZER_PAD_ID)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)

        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # máscaras
        src_pad_mask = ~make_pad_mask(
            src, TOKENIZER_PAD_ID)
        tgt_pad_mask = ~make_pad_mask(
            tgt, TOKENIZER_PAD_ID)
        tgt_causal = self._causal_mask(tgt.size(1), src.device)

        # encoder
        src_emb = self.pos_encoding(self.src_embedding(src))
        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)

        # decoder
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt))
        output = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_causal,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )

        return self.output_projection(output)

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(
            size, size, device=device), diagonal=1).bool()
        return mask
