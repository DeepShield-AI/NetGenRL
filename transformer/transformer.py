import torch
import torch.nn as nn
import math
from transformer.config import *

class GPTDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x, causal_mask):
        # Self-attention with causal mask
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask)
        x = x + attn_out  # residual

        # FFN
        h = self.ln2(x)
        ff_out = self.ff(h)
        x = x + ff_out  # residual

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x):
        B, S = x.shape[0], x.shape[1]
        # return positional embedding for concat: (B,S,d_model)
        return self.pe[:S].unsqueeze(0).expand(B, S, -1)


class TrafficTransformer(nn.Module):
    def __init__(self, field_vocab_sizes, field_emb_dim,
                 d_model, nhead, num_layers, ffn_dim, dropout, 
                 max_len, label_size, label_emb_dim,
                 max_seq_len, len_emb_dim):
        super().__init__()

        self.field_names = list(field_vocab_sizes.keys())
        self.field_vocab_sizes = field_vocab_sizes
        self.field_emb_dim = field_emb_dim
        self.max_seq_len = max_seq_len
        self.label_dim = label_size

        # per-field embedding
        self.field_embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab, field_emb_dim)
            for name, vocab in field_vocab_sizes.items()
        })
        
        
        self.field_offsets = {}       # 起始 ID
        current = 0
        for name, vocab in field_vocab_sizes.items():
            self.field_offsets[name] = current
            current += vocab
            
        # print(self.field_offsets)

        # field-type embedding
        self.field_type_emb = nn.Embedding(len(field_vocab_sizes), field_emb_dim)

        # label embedding
        self.label_emb = nn.Embedding(label_size, label_emb_dim)
        
        self.len_emb = nn.Embedding(max_seq_len,len_emb_dim)
        
        self.d_model = d_model
        

        # positional encoding (for concat)
        self.pos_enc = PositionalEncoding(d_model, max_len)

        # ------ CONCAT 开始 ------
        # token_emb_dim = field_emb_dim
        # field_type_emb_dim = field_emb_dim
        # label_emb_dim  (will be broadcast across seq)
        # pos_emb_dim = d_model
        # 最终 Transformer 输入维度：
        self.concat_dim = field_emb_dim + field_emb_dim + label_emb_dim + len_emb_dim + d_model
        # ------ CONCAT 结束 ------
        
        # self.combine_fc = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(self.concat_dim,self.d_model),
        #         nn.ReLU(True)
        #     ) for _ in range(self.label_dim)
        # ])

        # Transformer
        layer = nn.TransformerEncoderLayer(
            d_model=self.concat_dim,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.tr = nn.TransformerEncoder(layer, num_layers)

        self.ln = nn.LayerNorm(self.concat_dim)

        self.total_vocab = sum(field_vocab_sizes.values())
        self.out_layers = nn.ModuleDict({
            name: nn.Linear(self.concat_dim, vocab)
            for name, vocab in field_vocab_sizes.items()
        })

    def forward(self, seq, field_type_index, label, length, mask=None):
        B, S = seq.shape

        # ------------------------
        # 1. token embedding (per field)
        # ------------------------
        token_emb = torch.zeros(B, S, self.field_emb_dim, device=seq.device)

        for i, name in enumerate(self.field_names):
            # print(i,name)
            # print(field_type_index == i)
            batch_idx, pos_idx = (field_type_index == i).nonzero(as_tuple=True)
            if batch_idx.numel() == 0:
                continue
            token_ids = seq[batch_idx, pos_idx]
            # 将全局 token ID 映射为字段内局部 ID
            local_ids = token_ids - self.field_offsets[name]

            # 防越界
            if (local_ids < 0).any() or (local_ids >= self.field_vocab_sizes[name]).any():
                raise RuntimeError(f"Token ID 越界: 字段 {name}, 原始ID={token_ids}, local={local_ids}")
            field_emb = self.field_embeddings[name](local_ids)
            token_emb[batch_idx, pos_idx] = field_emb

        # ------------------------
        # 2. field type embedding
        # ------------------------
        # print(token_emb)
        field_type_emb = self.field_type_emb(field_type_index)  # (B,S,field_emb_dim)

        # print(field_type_emb)
        # ------------------------
        # 3. label embedding (broadcast to seq)
        # ------------------------
        lbl = self.label_emb(label)  # (B,label_emb_dim)
        lbl_emb = lbl.unsqueeze(1).expand(B, S, -1)  # (B,S,label_emb_dim)
        
        len_emb = self.len_emb(length).unsqueeze(1).expand(B, S, -1)

        # print(lbl_emb)
        # ------------------------
        # 4. positional embedding
        # ------------------------
        pos_emb = self.pos_enc(seq.new_zeros((B, S)))  # (B,S,d_model)

        # ------------------------
        # 5. CONCAT 所有 embedding
        # ------------------------
        x = torch.cat([token_emb, field_type_emb, lbl_emb, len_emb, pos_emb], dim=-1)
        # shape = (B, S, concat_dim)
        # fc_outputs = torch.stack([self.combine_fc[idx](combined) for idx in range(len(self.combine_fc))], dim=1)
        # indices = label.view(-1, 1, 1, 1).expand(-1, -1, fc_outputs.size(2), fc_outputs.size(3))

        # x = torch.gather(fc_outputs, dim=1, index=indices).squeeze(1)
        # print(x.shape)
        # ------------------------
        # 6. Transformer
        # ------------------------
        causal = torch.triu(torch.ones(S, S, device=seq.device), 1).bool()
        x = self.tr(x, mask=causal)
        x = self.ln(x)

        # ------------------------
        # 7. Output logits by field
        # ------------------------
        logits = torch.zeros(B, S, self.total_vocab, device=seq.device)

        for i, name in enumerate(self.field_names):
            batch_idx, pos_idx = (field_type_index == i).nonzero(as_tuple=True)
            if batch_idx.numel() == 0:
                continue
            logits[batch_idx, pos_idx, self.field_offsets[name]:self.field_offsets[name] + self.field_vocab_sizes[name]] = self.out_layers[name](x[batch_idx, pos_idx])

        if mask is not None:
            mask = mask.to(device=logits.device).bool()
            logits.masked_fill_(~mask.unsqueeze(-1), -1e9)
            # logits[~mask] = -1e9  # mask invalid positions

        return logits
    
    def top_k_logits(self,logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        min_val = v[:, -1].unsqueeze(1)    # shape [B,1]
        out[out < min_val] = -1e9
        return out
    
    @torch.no_grad()
    def generate(self, field_type, label, length, temperature=1.0, top_k = 0):

        self.eval()

        device = field_type.device
        B, S = field_type.shape

        label_token = self.field_offsets["label_token"] + label
        len_token   = self.field_offsets["len_token"]   + length
        label_sep_token = self.field_offsets["begin_token"]

        seq = torch.stack([
            label_token,
            len_token,
            torch.full_like(label, label_sep_token)
        ], dim=1).to(device)   # shape [B, 3]

        # 前 3 个字段类型
        # f_label = self.field_list.index("label_token")
        # f_len   = self.field_list.index("len_token")
        # f_sep   = self.field_list.index("sep_token")

        # field_types = torch.tensor(
        #     [[f_label, f_len, f_sep]] * B,
        #     dtype=torch.long,
        #     device=device
        # )   # (B,3)

        for t in range(3, S):

            logits = self(seq,field_type[:,:t],label,length)

            step_logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                # values, _ = torch.topk(step_logits, top_k)
                # min_val = values[-1]
                # step_logits[step_logits < min_val] = -1e9
                step_logits = self.top_k_logits(step_logits, top_k)

            # dataset 给出第 t 个 token 位置对应的字段类型（整数 id）
            fname_ids = field_type[:, t]    # shape (B,)

            next_tokens = []
            for i in range(B):

                fname = self.field_names[fname_ids[i].item()]

                start = self.field_offsets[fname]
                vocab = self.field_vocab_sizes[fname]

                slice_logits = step_logits[i, start:start+vocab]
                probs = torch.softmax(slice_logits, dim=-1)

                local_id = torch.multinomial(probs, 1).item()
                global_id = start + local_id

                next_tokens.append(global_id)

            next_tokens = torch.tensor(next_tokens, device=device)
            # print(next_tokens)
            # append
            seq = torch.cat([seq, next_tokens[:, None]], dim=1)
            # field_types = torch.cat([field_types, fname_ids[:, None]], dim=1)

        return seq