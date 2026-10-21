"""Train, select, and generate traffic with the byte-level Transformer."""

import argparse
import json
import os
import sys

import numpy as np
import ot
import torch

from config import (DEVICE, MAX_SEQ_LEN, EMBED_DIM, NUM_HEADS,
                    NUM_LAYERS, FFN_DIM, DROPOUT)
from dataset import FlowByteDataset
from tokenizer import ByteTokenizer, TOKEN_SEP, TOKEN_EOS
from train import train
from transformer import TrafficTransformer


SERIES_ATTRS = ("time", "pkt_len", "flags", "ttl")
META_ATTRS = ("src_ip", "dst_ip", "src_port", "dst_port")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model", "baseline-transformer")
RESULT_DIR = os.path.join(PROJECT_ROOT, "result", "baseline-transformer")
sys.path.insert(0, PROJECT_ROOT)


def discover_labels(data_dir):
    files = sorted(name for name in os.listdir(data_dir) if name.endswith(".json"))
    labels = {os.path.splitext(name)[0]: index for index, name in enumerate(files)}
    if not labels:
        raise ValueError(f"No JSON label files found under {data_dir}")
    return labels


def make_model():
    tokenizer = ByteTokenizer()
    return TrafficTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=EMBED_DIM,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ffn_dim=FFN_DIM,
        dropout=DROPOUT,
        max_len=MAX_SEQ_LEN,
    )


def load_flows(data_dir, labels, limit=None):
    flows_by_label = {}
    for label in labels:
        path = os.path.join(data_dir, f"{label}.json")
        with open(path, "r") as stream:
            flows = json.load(stream)
        flows_by_label[label] = flows[:limit] if limit else flows
    return flows_by_label


def flow_to_metric_sequence(flow, max_seq_len):
    """Match check_bestmodel's normalized [series, metadata] representation."""
    meta = [flow[name] for name in META_ATTRS]
    sequence = []
    for packet in flow["series"][:max_seq_len]:
        sequence.append([packet[name] for name in SERIES_ATTRS] + meta)
    return sequence


def normalized_metric_data(flows, bins, max_seq_len):
    result = []
    for flow in flows:
        sequence = flow_to_metric_sequence(flow, max_seq_len)
        for col, name in enumerate(SERIES_ATTRS + META_ATTRS):
            scale = bins[name]["intervals"][-1][1]
            for row in sequence:
                if scale == 0:
                    row[col] = 0.0
                    continue
                row[col] /= scale
        result.append(sequence)
    return result


def padded_metric_array(sequences, max_seq_len):
    width = len(SERIES_ATTRS) + len(META_ATTRS)
    values = np.full((len(sequences), max_seq_len, width), np.nan, dtype=np.float64)
    for index, sequence in enumerate(sequences):
        if sequence:
            values[index, :len(sequence)] = sequence[:max_seq_len]
    return np.nan_to_num(values, nan=1.0)


def ot_score(real_flows, generated_flows, bins, max_seq_len):
    real = padded_metric_array(normalized_metric_data(real_flows, bins, max_seq_len), max_seq_len)
    fake = padded_metric_array(normalized_metric_data(generated_flows, bins, max_seq_len), max_seq_len)
    costs = np.linalg.norm(real[:, None] - fake[None, :], axis=(-2, -1))
    return float(ot.emd2([], [], costs))


def build_model_from_checkpoint(path):
    model = make_model()
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["state"])
    return model.to(DEVICE).eval()


@torch.no_grad()
def generate_flows(model, label, label_id, real_flows, batch_size, max_packets, temperature, top_k):
    """Generate one structurally complete byte sequence per real flow."""
    results = [None] * len(real_flows)
    by_length = {}
    for index, real_flow in enumerate(real_flows):
        length = min(len(real_flow["series"]), max_packets)
        if length < 1:
            raise ValueError(f"{label} flow {index} has no packets")
        by_length.setdefault(length, []).append(index)

    for packet_count, indexes in by_length.items():
        for start in range(0, len(indexes), batch_size):
            group = indexes[start:start + batch_size]
            ids = torch.tensor([[label_id, TOKEN_SEP]] * len(group), device=DEVICE)
            # Header is 12 bytes; each packet is 13 bytes and a separator.
            for position in range(12 + 14 * packet_count):
                if position >= 12 and (position - 12) % 14 == 13:
                    next_token = torch.full((len(group), 1), TOKEN_SEP,
                                            dtype=torch.long, device=DEVICE)
                else:
                    mask = torch.ones_like(ids, dtype=torch.bool)
                    logits = model(ids, mask)[:, -1, :256] / temperature
                    if top_k < 256:
                        values, selected = torch.topk(logits, top_k)
                        filtered = torch.full_like(logits, float("-inf"))
                        filtered.scatter_(1, selected, values)
                        logits = filtered
                    next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                ids = torch.cat((ids, next_token), dim=1)
            ids = torch.cat((ids, torch.full((len(group), 1), TOKEN_EOS,
                                             dtype=torch.long, device=DEVICE)), dim=1)
            for index, sequence in zip(group, ids.tolist()):
                results[index] = decode_structured_flow(sequence, packet_count)
    return results


def decode_structured_flow(sequence, packet_count):
    """Decode the fixed byte layout sampled above without recovery heuristics."""
    import struct

    raw = bytes(sequence[2:14])
    src_ip, dst_ip, src_port, dst_port = struct.unpack(">IIHH", raw)
    flow = {"src_ip": src_ip, "dst_ip": dst_ip,
            "src_port": src_port, "dst_port": dst_port, "series": []}
    for packet_index in range(packet_count):
        start = 14 + packet_index * 14
        payload = bytes(sequence[start:start + 13])
        time_value, packet_length, direction, flags, ttl = struct.unpack(">dHBBB", payload)
        flow["series"].append({
            "time": min(10000.0, max(0.0, time_value)) if np.isfinite(time_value) else 0.0,
            "pkt_len": min(1500, packet_length) * (-1 if direction == 1 else 1),
            "flags": flags,
            "ttl": ttl,
        })
    return flow


def save_like_data_save(label, flows, output_dir):
    """Adapt decoded flows to post_process.data_save.save_data's JSON shape."""
    os.makedirs(output_dir, exist_ok=True)
    converted = []
    for flow in flows:
        packets = flow.get("series", [])
        if not packets:
            continue
        # save_data stores metadata from the first packet row and packet fields
        # in series, so retain its layout while keeping the Transformer values.
        seq = []
        metadata = [flow[name] for name in META_ATTRS]
        for packet in packets:
            seq.append([packet.get(name, 0) for name in SERIES_ATTRS] + metadata)
        converted.append(seq)

    # Use the repository's established serializer and field layout.
    from post_process.data_save import save_data
    save_data(label, converted, list(META_ATTRS), list(SERIES_ATTRS), output_dir + os.sep)


def run(args):
    data_dir = os.path.abspath(args.data)
    dataset_name = os.path.basename(data_dir.rstrip(os.sep))
    model_dir = os.path.join(MODEL_DIR, dataset_name)
    result_dir = os.path.join(RESULT_DIR, dataset_name)
    if args.epochs < 1 or args.patience < 0 or args.eval_samples < 1 or args.batch_size < 1:
        raise ValueError("epochs, eval-samples, and batch-size must be positive; patience cannot be negative")
    if args.max_packets < 1 or 15 + 14 * args.max_packets > MAX_SEQ_LEN:
        raise ValueError("max-packets exceeds the model's position capacity")
    if args.temperature <= 0 or not 1 <= args.top_k <= 256:
        raise ValueError("temperature must be positive and top-k between 1 and 256")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    labels = discover_labels(data_dir)
    if len(labels) > 256:
        raise ValueError("Byte-token labels support at most 256 classes")
    bins_path = args.bins or os.path.join(PROJECT_ROOT, "bins", f"bins_{dataset_name}.json")
    with open(bins_path, "r") as stream:
        bins = json.load(stream)
    dataset = FlowByteDataset(data_dir, labels, MAX_SEQ_LEN)
    if len(dataset) < 2:
        raise ValueError(f"Need at least two usable flows; found {len(dataset)}")
    print(f"Training on {len(dataset)} flows using {DEVICE}")
    final_path = os.path.join(model_dir, "traffic_transformer.pth")
    completed_epochs = train(
        data_dir,
        args.epochs,
        final_path,
        patience=args.patience,
        checkpoint_dir=model_dir,
        checkpoint_interval=50,
        labels=labels,
    )
    completed_epochs = 150

    real_for_eval = load_flows(data_dir, labels, limit=args.eval_samples)
    checkpoint_epochs = list(range(50, completed_epochs + 1, 50))
    if completed_epochs % 50 or not checkpoint_epochs:
        checkpoint_epochs.append(completed_epochs)
        last_path = os.path.join(model_dir, f"epoch_{completed_epochs}.pth")
        torch.save(torch.load(final_path, map_location="cpu"), last_path)

    best_epoch, best_score = None, float("inf")
    scores = {}
    for epoch in checkpoint_epochs:
        path = os.path.join(model_dir, f"epoch_{epoch}.pth")
        model = build_model_from_checkpoint(path)
        epoch_score = 0.0
        for label, real_flows in real_for_eval.items():
            fake_flows = generate_flows(
                model, label, labels[label], real_flows, args.batch_size,
                args.max_packets, args.temperature, args.top_k
            )
            score = ot_score(real_flows, fake_flows, bins, args.max_packets)
            print(f"epoch {epoch} {label} OT: {score:.6f}")
            epoch_score += score
        scores[str(epoch)] = epoch_score
        print(f"epoch {epoch} total OT: {epoch_score:.6f}")
        if epoch_score < best_score:
            best_epoch, best_score = epoch, epoch_score

    if best_epoch is None:
        raise RuntimeError("No checkpoints were available for evaluation")
    best_checkpoint = os.path.join(model_dir, f"epoch_{best_epoch}.pth")
    with open(os.path.join(model_dir, "evaluation.json"), "w") as stream:
        json.dump({"best_epoch": best_epoch, "best_ot": best_score, "scores": scores}, stream, indent=2)
    model = build_model_from_checkpoint(best_checkpoint)
    real_for_generation = load_flows(data_dir, labels)
    for label, real_flows in real_for_generation.items():
        generated = generate_flows(
            model, label, labels[label], real_flows, args.batch_size,
            args.max_packets, args.temperature, args.top_k
        )
        save_like_data_save(label, generated, result_dir)
        print(f"Generated {len(generated)}/{len(real_flows)} flows for {label}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.path.join(PROJECT_ROOT, "data-json", "iscx_small"))
    parser.add_argument("--bins", default=None,
                        help="defaults to bins/bins_<dataset>.json")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=0,
                        help="early-stop patience; 0 runs all requested epochs")
    parser.add_argument("--eval-samples", type=int, default=100,
                        help="maximum real flows per label used for checkpoint OT evaluation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-packets", type=int, default=16,
                        help="packet count cap for generation and OT, matching config.json")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
