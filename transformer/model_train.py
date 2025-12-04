import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from transformer.dataset import FlattenTransformerDataset, collate_fn
from transformer.transformer import TrafficTransformer
from transformer.config import *
import time

def compute_loss(logits, tgt, ftype, field_names, field_vocab_sizes, weights):
    B, T, V = logits.shape
    # print(B,T,V)
    device = logits.device

    offsets = []
    cur = 0
    for sz in field_vocab_sizes.values():
        offsets.append(cur)
        cur += sz
    offsets = torch.tensor(offsets, device=device)   # shape = (F,)
    # 展开成一维，方便索引
    logits = logits.reshape(B*T, V)
    tgt = tgt.reshape(B*T)
    ftype = ftype.reshape(B*T)
    weights_expanded = weights.repeat_interleave(T)

    # --- 为每个 token 挑选自己的 vocab 切片 ---
    losses = []
    

    for i,f in enumerate(field_names):
        # 找出属于字段 f 的所有 token
        mask = (ftype == i)
        if mask.sum() == 0:
            continue  # 某些字段可能在当前 batch 中没有出现

        # logits 的切片范围
        start = offsets[i].item()
        end = start + field_vocab_sizes[f]

        # 取出对应字段的局部 logits
        
        logits_f = logits[mask, start:end]   # (N_f, vocab_f)
        tgt_f = tgt[mask] - start            # (N_f)
        w_f = weights_expanded[mask]
        
        # print(logits_f)
        # print(tgt_f)
        
        logp_f = F.log_softmax(logits_f, dim=-1)

        # 使用 NLLLoss（手动加权）
        loss_f = F.nll_loss(
            logp_f,
            tgt_f,
            reduction="none"
        )
        
        # 计算交叉熵损失
        loss_f = F.cross_entropy(logits_f, tgt_f)
        weighted_loss_f = (loss_f * w_f).sum()
        losses.append(weighted_loss_f)

    # 平均所有字段的损失（等权）
    if len(losses) == 0:
        return torch.tensor(0.0, device=device)

    return sum(losses) / len(losses)


def train(ds, model, epochs, model_folder,checkpoint,batch_size, device, patience=10, min_delta=1e-4):

    # -------- Train/Val split --------
    val_size = max(1, int(0.1 * len(ds)))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # train_dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # -------- Early Stopping --------
    best_val_loss = float("inf")
    last_val_loss = float("inf")
    bad_epochs = 0
    # best_path = save_path.replace(".pth", "_best.pth")
    best_path = f"{model_folder}/transformer_best.pth"

    # ======================================================
    # Epoch Loop
    # ======================================================
    for ep in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        total = 0

        # ---------------- Train Loop ----------------
        for i,batch in enumerate(train_dl):

            seq   = batch["seq"].to(device)
            ftype = batch["field_type"].to(device)
            mask  = batch["mask"].to(device)
            label = batch["label"].to(device)
            tgt   = batch["target"].to(device)
            length = batch["len"].to(device)
            weight = batch["weight"].to(device)

            logits = model(seq, ftype, label,length, mask)

            loss = compute_loss(logits,tgt,ftype,model.field_names,model.field_vocab_sizes, weight)
            
            # print("batch",i )

            opt.zero_grad()
            loss.backward()

            # gradient clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()

            total += loss.item()
            
            if i % 50 == 0:
                print(f"[Train] Epoch {ep} step {i} loss={loss.item():.4f}")
        end = time.perf_counter()
        avg_train_loss = total / i
        print(f"[Train] Epoch {ep} average loss {avg_train_loss:.4f}, cost time {end-start}.")

        # ---------------- Validation Loop ----------------
        model.eval()
        val_total = 0
        val_step = 0

        with torch.no_grad():
            for batch in val_dl:

                seq   = batch["seq"].to(device)
                ftype = batch["field_type"].to(device)
                mask  = batch["mask"].to(device)
                label = batch["label"].to(device)
                tgt   = batch["target"].to(device)
                length = batch["len"].to(device)
                weight = batch["weight"].to(device)

                logits = model(seq, ftype, label, length, mask)
                loss = compute_loss(logits, tgt, ftype, model.field_names, model.field_vocab_sizes, weight)

                val_total += loss.item()
                val_step += 1

        avg_val_loss = val_total / val_step
        print(f"[Valid] Epoch {ep} val loss = {avg_val_loss:.4f}")
        
        if ep % checkpoint == 0:
            checkpoint_path = f"{model_folder}/transformer_{ep}.pth"
            torch.save({"state": model.state_dict()}, checkpoint_path)
            print(f"[Checkpoint] Model checkpoint saved to: {checkpoint_path}")

        # ---------------- Early Stopping ----------------
        if avg_val_loss + min_delta < best_val_loss:
            best_val_loss = min(avg_val_loss,best_val_loss)
            last_val_loss = avg_val_loss
        # if avg_train_loss + min_delta < best_val_loss:
        #     best_val_loss = min(avg_train_loss,best_val_loss)
        #     last_val_loss = avg_train_loss
            bad_epochs = 0
            torch.save({"state": model.state_dict()}, best_path)
            print(f"[EarlyStopping] New best model saved to: {best_path}")
        else:
            bad_epochs += 1
            print(f"[EarlyStopping] No improvement for {bad_epochs} epochs")

        if bad_epochs >= patience:
            print(f"\n[EarlyStopping] Stop at Epoch {ep}. Best val loss={best_val_loss:.4f}")
            break

    # -------- Save final model --------
    # torch.save({"state": model.state_dict()}, save_path)
    # print("Final model saved:", save_path)
    print("Best model saved:", best_path)


# ======================================================
# CLI
# ======================================================
def model_train(label_dict, dataset, json_folder, bins_folder, model_folder,
                port_attrs,ip_attrs, sery_attrs, 
                model_paras):
    # p = argparse.ArgumentParser()
    # p.add_argument("--data", type=str, required=True)
    # p.add_argument("--epochs", type=int, default=30)
    # p.add_argument("--save", type=str, default="./model/traffic_transformer.pth")
    # args = p.parse_args()
    
    label_dim = len(label_dict)
    batch_size = model_paras['batch_size']
    epochs = model_paras['epoch']
    max_pkt_len = model_paras['max_seq_len']
    # series_word_vec_size = model_paras['series_word_vec_size']
    # meta_word_vec_size = model_paras['meta_word_vec_size']
    # n_critic = model_paras['n_critic']
    # n_roll = model_paras['n_roll']
    checkpoint = model_paras['checkpoint']
    # pre_trained_generator_epoch = model_paras['pre_trained_generator_epoch']
    # pre_trained_discriminator_epoch = model_paras['pre_trained_discriminator_epoch']
    
    # seq_dim = len(sery_attrs) + len(port_attrs) + len(ip_attrs)
    
    data_folder = f'./{json_folder}/{dataset}/'
    bins_file_name = f'./{bins_folder}/bins_{dataset}.json'
    # wordvec_file_name = f'./{wordvec_folder}/word_vec_{dataset}.json'
    model_folder_name = f'./{model_folder}/{dataset}/'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ds = FlattenTransformerDataset(
        data_folder=data_folder,
        bins_file=bins_file_name,
        class_mapping=label_dict,
        max_seq_len=max_pkt_len,
        port_attrs=port_attrs,
        ip_attrs=ip_attrs,
        sery_attrs=sery_attrs
    )
    
    print(ds.field_vocab_sizes)
    
    max_len = cal_max_len(port_attrs,max_pkt_len,sery_attrs,ip_attrs)
    model = TrafficTransformer(
        field_vocab_sizes=ds.field_vocab_sizes,
        field_emb_dim=EMBED_DIM,
        d_model=EMBED_DIM,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ffn_dim=FFN_DIM,
        dropout=DROPOUT,
        max_len=max_len,
        label_size=label_dim,
        label_emb_dim=LABEL_EMB_DIM,
        max_seq_len=max_pkt_len,
        len_emb_dim=LEN_EMB_DIM
    ).to(device)

    print("Device:", device)
    
    train(ds, model, epochs, model_folder_name, checkpoint, batch_size, device)

