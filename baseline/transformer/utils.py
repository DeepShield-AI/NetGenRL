# utils.py

import torch
import torch.nn as nn
from tokenizer import TOKEN_EOS

def compute_loss(logits, ids, attn_mask):
    # shift for next-token prediction
    logits = logits[:, :-1, :]
    targets = ids[:, 1:]
    mask = attn_mask[:, 1:]

    loss_fct = nn.CrossEntropyLoss(reduction="none")

    B, S, V = logits.shape
    loss = loss_fct(logits.reshape(B * S, V), targets.reshape(-1))
    loss = loss.reshape(B, S)

    loss = loss * mask.float()
    denom = mask.sum()

    return loss.sum() / (denom + 1e-6)
