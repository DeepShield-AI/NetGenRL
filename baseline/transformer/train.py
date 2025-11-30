import torch
from torch.utils.data import DataLoader, random_split
import argparse

from tokenizer import ByteTokenizer
from dataset import FlowByteDataset, collate_fn
from transformer import TrafficTransformer
from utils import compute_loss
from config import *

def train(data_path, epochs, save_path, patience=5, min_delta=1e-4):
    tokenizer = ByteTokenizer()
    ds = FlowByteDataset(data_path, LABELS, MAX_SEQ_LEN)

    # ---- 自动划分 train/val ----
    val_size = max(1, int(0.1 * len(ds)))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # ---- 模型 ----
    model = TrafficTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=EMBED_DIM,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ffn_dim=FFN_DIM,
        dropout=DROPOUT,
        max_len=MAX_SEQ_LEN
    ).to(DEVICE)

    print(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # ---- Early Stopping 参数 ----
    best_val_loss = float("inf")
    bad_epochs = 0
    best_path = save_path.replace(".pth", "_best.pth")

    for ep in range(1, epochs + 1):
        model.train()
        total = 0
        step = 0

        # ----------- Train loop -----------
        for ids, attn in train_dl:
            ids, attn = ids.to(DEVICE), attn.to(DEVICE)

            logits = model(ids, attn)
            loss = compute_loss(logits, ids, attn)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += loss.item()
            step += 1

            if step % 50 == 0:
                print(f"[Train] Epoch {ep} step {step} loss={loss.item():.4f}")

        avg_train_loss = total / step
        print(f"[Train] Epoch {ep} average loss {avg_train_loss:.4f}")

        # ----------- Validation loop -----------
        model.eval()
        val_total = 0
        val_step = 0

        with torch.no_grad():
            for ids, attn in val_dl:
                ids, attn = ids.to(DEVICE), attn.to(DEVICE)
                logits = model(ids, attn)
                loss = compute_loss(logits, ids, attn)
                val_total += loss.item()
                val_step += 1

        avg_val_loss = val_total / val_step
        print(f"[Valid] Epoch {ep} val loss = {avg_val_loss:.4f}")

        # ----------- Early Stopping 判断 -----------
        if avg_val_loss + min_delta < best_val_loss:
            best_val_loss = avg_val_loss
            bad_epochs = 0
            torch.save({"state": model.state_dict()}, best_path)
            print(f"[EarlyStopping] New best model saved to {best_path}")
        else:
            bad_epochs += 1
            print(f"[EarlyStopping] No improvement for {bad_epochs} epochs")

        if bad_epochs >= patience:
            print(f"\n[EarlyStopping] Stop training at epoch {ep}. "
                  f"Best val loss = {best_val_loss:.4f}")
            break

    # ---- 保存最终模型（不一定是最好）----
    torch.save({"state": model.state_dict()}, save_path)
    print("Final model saved:", save_path)
    print("Best model saved:", best_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--save", type=str, default="./model/traffic_transformer.pth")
    args = p.parse_args()

    train(args.data, args.epochs, args.save)
