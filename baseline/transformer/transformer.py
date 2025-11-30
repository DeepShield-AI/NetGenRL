# model.py

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TrafficTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, ffn_dim, dropout, max_len):
        super().__init__()

        self.embed_dim = d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, ids, attn_mask):
        B, S = ids.shape

        x = self.embed(ids) * math.sqrt(self.embed_dim)
        x = self.pos(x)

        causal = torch.triu(torch.ones(S, S, device=ids.device), 1).bool()
        key_pad_mask = ~attn_mask

        x = self.tr(x, mask=causal, src_key_padding_mask=key_pad_mask)

        x = self.ln(x)
        return self.fc(x)
