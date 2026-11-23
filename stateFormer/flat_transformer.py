"""Attribute-token causal Transformer with independent input noise per position."""
import torch
from torch import nn
from torch.nn import functional as F


class FlatTransGenerator(nn.Module):
    model_type = "flat_transformer_v1"

    def __init__(self, label_dim, seq_dim, max_seq_len, x_list, device=None,
                 hidden_dim=512, embedding_dim=128, noise_dim=32,
                 transformer_layers=4, nhead=8, dropout=0.1, noise_scale=1.0):
        super().__init__()
        if seq_dim != len(x_list) or not x_list or min(x_list) < 1:
            raise ValueError("One positive vocabulary size is required per field")
        if min(label_dim, max_seq_len, hidden_dim, embedding_dim, noise_dim,
               transformer_layers, nhead) < 1 or hidden_dim % nhead:
            raise ValueError("Invalid dimensions or attention head count")
        if noise_scale < 0:
            raise ValueError("noise_scale must be nonnegative")
        self.label_dim = label_dim
        self.seq_dim = seq_dim
        self.max_seq_len = max_seq_len  # packets, not attribute tokens
        self.max_tokens = max_seq_len * seq_dim
        self.x_list = list(x_list)
        self.output_sizes = [n + 1 for n in self.x_list]
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.noise_scale = noise_scale
        self._model_config = dict(
            label_dim=label_dim, seq_dim=seq_dim, max_seq_len=max_seq_len,
            x_list=list(x_list), hidden_dim=hidden_dim,
            embedding_dim=embedding_dim, noise_dim=noise_dim,
            transformer_layers=transformer_layers, nhead=nhead,
            dropout=dropout, noise_scale=noise_scale,
        )
        self.value_embeddings = nn.ModuleList([
            nn.Embedding(n + 1, embedding_dim, padding_idx=0) for n in x_list
        ])
        self.value_projection = nn.Linear(embedding_dim, hidden_dim)
        self.bos = nn.Parameter(torch.zeros(hidden_dim))
        self.field_embedding = nn.Embedding(seq_dim, hidden_dim)
        self.packet_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.label_embedding = nn.Embedding(label_dim, hidden_dim)
        self.length_condition = nn.Sequential(
            nn.Linear(max_seq_len, 64), nn.ReLU(), nn.Linear(64, hidden_dim)
        )
        self.noise_projection = nn.Linear(noise_dim, hidden_dim, bias=False)
        self.input_norm = nn.LayerNorm(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="relu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, transformer_layers, enable_nested_tensor=False
        )
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, n + 1) for n in x_list
        ])
        if device is not None:
            self.to(device)

    def model_config(self):
        return dict(self._model_config)

    @staticmethod
    def shift_targets(sequences):
        """Packet-major flattening, shifted by ONE ATTRIBUTE (not one packet)."""
        targets = sequences.flatten(1)
        inputs = torch.cat([targets.new_zeros((len(targets), 1)), targets[:, :-1]], dim=1)
        return inputs

    def _hidden(self, labels, lengths, inputs, noise=None):
        if inputs.ndim != 2 or not 1 <= inputs.shape[1] <= self.max_tokens:
            raise ValueError("Expected [batch, attribute_positions] within max_tokens")
        batch, tokens = inputs.shape
        device = inputs.device
        lengths = lengths.to(device=device, dtype=torch.long)
        if lengths.shape != (batch,) or ((lengths < 1) | (lengths > self.max_seq_len)).any():
            raise ValueError("Lengths must be valid packet counts")
        if torch.any(inputs[:, 0] != 0):
            raise ValueError("Position zero must contain BOS=0")
        if labels.ndim == 2:
            if labels.shape != (batch, self.label_dim):
                raise ValueError("One-hot label dimensions do not match model")
            label_ids = labels.argmax(dim=1)
        elif labels.shape == (batch,):
            label_ids = labels.long()
        else:
            raise ValueError("Expected label IDs or one-hot labels")
        label_ids = label_ids.to(device)
        positions = torch.arange(tokens, device=device)
        # Input position q contains y[q-1]. Its VALUE embedding belongs to the
        # previous field, whereas field_embedding describes the TARGET field q.
        value = self.bos.new_zeros((batch, tokens, self.hidden_dim))
        value[:, 0] = self.bos
        for field, embedding in enumerate(self.value_embeddings):
            if field + 1 >= tokens:
                continue
            previous_positions = torch.arange(field + 1, tokens, self.seq_dim, device=device)
            value[:, previous_positions] = self.value_projection(
                embedding(inputs[:, previous_positions])
            )
        if noise is None:
            noise = torch.randn(batch, tokens, self.noise_dim, device=device, dtype=value.dtype)
        if noise.shape != (batch, tokens, self.noise_dim) or noise.device != device:
            raise ValueError("Noise must match [batch, positions, noise_dim] and device")
        length_one_hot = F.one_hot(lengths - 1, self.max_seq_len).to(value.dtype)
        value = (
            value
            + self.field_embedding(positions % self.seq_dim)[None]
            + self.packet_embedding(positions // self.seq_dim)[None]
            + self.label_embedding(label_ids)[:, None]
            + self.length_condition(length_one_hot)[:, None]
            + self.noise_scale * self.noise_projection(noise)
        )
        causal = torch.triu(torch.ones(tokens, tokens, device=device, dtype=torch.bool), diagonal=1)
        padding = positions[None] >= (lengths * self.seq_dim)[:, None]
        return self.transformer(self.input_norm(value), mask=causal, src_key_padding_mask=padding)

    @staticmethod
    def _exclude_zero(logits):
        logits[..., 0] = torch.finfo(logits.dtype).min
        return logits

    def forward(self, labels, lengths, inputs, noise=None):
        hidden = self._hidden(labels, lengths, inputs, noise)
        # Each head only sees positions predicting that field. No giant
        # [batch, max_tokens, sum(vocabularies)] allocation is constructed.
        return {"logits": tuple(
            self._exclude_zero(head(hidden[:, field::self.seq_dim]))
            for field, head in enumerate(self.output_heads)
        )}

    def next_logits(self, labels, lengths, inputs, noise):
        hidden = self._hidden(labels, lengths, inputs, noise)
        field = (inputs.shape[1] - 1) % self.seq_dim
        return self._exclude_zero(self.output_heads[field](hidden[:, -1]))

    @torch.no_grad()
    def sample(self, batch_size, label, length, x=None, noise=None):
        if self.training:
            raise ValueError("Call eval() before sampling")
        device = self.bos.device
        if len(label) != batch_size or len(length) != batch_size:
            raise ValueError("Batch size and conditioning dimensions disagree")
        generated = torch.zeros(batch_size, self.max_tokens, dtype=torch.long, device=device)
        given = 0
        if x is not None:
            if x.ndim != 3 or x.shape[0] != batch_size or x.shape[2] != self.seq_dim:
                raise ValueError("Prefix must have [batch, packets, fields] shape")
            if x.shape[1] > self.max_seq_len:
                raise ValueError("Prefix exceeds configured packet length")
            for field, bins in enumerate(self.x_list):
                if ((x[:, :, field] < 1) | (x[:, :, field] > bins)).any():
                    raise ValueError("Prefix contains padding or invalid bin IDs")
            given = x.shape[1] * self.seq_dim
            generated[:, :given] = x.to(device).flatten(1)
        if noise is None:
            noise = torch.randn(batch_size, self.max_tokens, self.noise_dim, device=device)
        if noise.shape != (batch_size, self.max_tokens, self.noise_dim) or noise.device != device:
            raise ValueError("Sampling requires one full, fixed positional noise bank")
        # Reuse the SAME noise prefix on every recomputation of history.
        for position in range(given, self.max_tokens):
            inputs = torch.cat([generated.new_zeros((batch_size, 1)), generated[:, :position]], dim=1)
            logits = self.next_logits(label, length, inputs, noise[:, :position + 1])
            generated[:, position] = F.softmax(logits, dim=-1).multinomial(1).squeeze(1)
        return generated.reshape(batch_size, self.max_seq_len, self.seq_dim)
