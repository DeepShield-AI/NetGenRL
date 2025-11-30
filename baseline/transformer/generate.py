# generate.py

import argparse
import os
import torch

from tokenizer import ByteTokenizer, TOKEN_SEP, TOKEN_EOS
from transformer import TrafficTransformer
from config import *
from decoder import decode_flow, save_flow_json


@torch.no_grad()
def generate_batch(model, prompts, max_new=512, temperature=1.0, top_k=50):
    """
    prompts: List[List[int]]
    returns: List[List[int]]
    """
    batch_size = len(prompts)

    # pad prompts to the same length
    max_len = max(len(p) for p in prompts)
    cur = torch.full((batch_size, max_len), TOKEN_EOS, dtype=torch.long, device=DEVICE)
    for i, p in enumerate(prompts):
        if len(p) > 0:
            cur[i, :len(p)] = torch.tensor(p, device=DEVICE)

    finished = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)

    for _ in range(max_new):
        attn_mask = torch.ones_like(cur, dtype=torch.bool)
        logits = model(cur, attn_mask)

        logits = logits[:, -1, :] / temperature  # (B, vocab)

        # top-k filtering
        if top_k > 0:
            vals, idx = torch.topk(logits, top_k)
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(1, idx, vals)
            logits = mask

        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, 1).squeeze(1)  # (B,)

        # sequences already finished: force EOS
        next_tokens = torch.where(finished, torch.tensor(TOKEN_EOS, device=DEVICE), next_tokens)

        # append next tokens
        cur = torch.cat([cur, next_tokens.unsqueeze(1)], dim=1)

        # update finished mask
        finished |= (next_tokens == TOKEN_EOS)

        if finished.all():
            break

    # convert to python lists
    return [seq.tolist() for seq in cur]


def parse_prompt(hex_str):
    """Convert hex string (with | removed) to byte list"""
    hex_str = hex_str.replace("|", "")
    if len(hex_str) % 2 == 1:
        hex_str = "0" + hex_str
    bs = bytes.fromhex(hex_str)
    return list(bs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to checkpoint")
    parser.add_argument("--prompt", required=True, help="Hex string prompt")
    parser.add_argument("--num_flows", type=int, default=1, help="Number of flows to generate")
    parser.add_argument("--max_new", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--save", type=str, default="output")
    args = parser.parse_args()

    # os.makedirs(args.save, exist_ok=True)
    
    if args.prompt not in LABELS:
        raise ValueError(f"Prompt label '{args.prompt}' not in known labels: {list(LABELS.keys())}")
    prompt = [LABELS[args.prompt], TOKEN_SEP]

    tok = ByteTokenizer()

    # build model
    model = TrafficTransformer(
        vocab_size=tok.vocab_size,
        d_model=EMBED_DIM,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ffn_dim=FFN_DIM,
        dropout=DROPOUT,
        max_len=MAX_SEQ_LEN
    )

    ckpt = torch.load(args.model, map_location=DEVICE)
    model.load_state_dict(ckpt["state"])
    model = model.to(DEVICE)
    model.eval()

    # build prompts
    prompts = []
    # if args.prompt:
    #     base_prompt = parse_prompt(args.prompt)
    # else:
    #     base_prompt = [TOKEN_EOS]

    for _ in range(args.num_flows):
        prompts.append(prompt)

    # generate
    outs = generate_batch(
        model,
        prompts,
        max_new=args.max_new - len(prompt),
        temperature=args.temperature,
        top_k=args.top_k
    )

    # decode & save
    flows = []
    for i, seq in enumerate(outs):
        flow = decode_flow(seq)
        flows.append(flow)
        # save_path = os.path.join(args.save_dir, f"flow_{i}.json")
    save_flow_json(flows, f"{args.save}")
    print(f"Saved: {args.save}")
