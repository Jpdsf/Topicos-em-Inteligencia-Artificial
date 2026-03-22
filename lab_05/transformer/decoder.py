import torch
import torch.nn as nn


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        y: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ) -> torch.Tensor:
        # masked self-attention
        attn1, _ = self.self_attn(
            y, y, y, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        y = self.norm1(y + self.dropout(attn1))

        # cross-attention
        attn2, _ = self.cross_attn(
            y, memory, memory, key_padding_mask=memory_key_padding_mask)
        y = self.norm2(y + self.dropout(attn2))

        ffn_out = self.ffn(y)
        y = self.norm3(y + self.dropout(ffn_out))
        return y


class Decoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        y: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ) -> torch.Tensor:
        for layer in self.layers:
            y = layer(
                y, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return y
