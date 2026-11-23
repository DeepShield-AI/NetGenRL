"""Independent complete-ChiMerge -> flat Transformer -> evaluation -> JSON entry."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from pre_process.ChiMerge import chimerge
from stateFormer.dataset import CustomDataset
from stateFormer.flat_transformer import FlatTransGenerator
from stateFormer.flat_train import train_flat
from post_process.flat_pipeline import select_flat_checkpoint, generate_flat_data

ROOT = Path(__file__).resolve().parent


def run_flat(config, device=None):
    os.chdir(ROOT)
    if config.get("model_type") != FlatTransGenerator.model_type:
        raise ValueError("Use an explicit flat_transformer_v1 config")
    if config.get("binning", {}).get("method", "chimerge") != "chimerge":
        raise ValueError("This baseline requires complete original ChiMerge")
    paths, params, attrs = config["path"], config["model_paras"], config["attributes"]
    if params["epoch"] <= 0 or params["checkpoint"] <= 0 or params["epoch"] % params["checkpoint"]:
        raise ValueError("epoch must be positive and divisible by checkpoint")
    if params["expand_times"] < 0 or params["batch_size"] <= 0:
        raise ValueError("Invalid generation multiplier or batch size")
    dataset_name = paths["dataset"]
    bins_file = Path(paths["bins_folder"]) / f"bins_{dataset_name}.json"
    model_dir = Path(paths["model_folder"]) / dataset_name
    output_dir = Path(paths["result_folder"]) / dataset_name
    for path in (bins_file, model_dir, output_dir):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Fresh flat run refuses existing output: {path}")
    files = sorted((Path(paths["json_folder"]) / dataset_name).glob("*.json"))
    labels = {p.name.split(".")[0]: i for i, p in enumerate(files)}
    if not labels or len(labels) != len(files):
        raise ValueError("Empty input or ambiguous class filenames")
    seed = config.get("seed", 20261120)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    sery = list(attrs["sery_attribute"])
    ports, ips = list(attrs["port_attribute"]), list(attrs["ip_attribute"])
    fields = sery + ports + ips
    attr_params = {**attrs["sery_attribute"], **attrs["port_attribute"], **attrs["ip_attribute"]}
    model_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    bins_file.parent.mkdir(parents=True, exist_ok=True)
    (model_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    print("Complete ChiMerge ...", flush=True)
    chimerge(dataset_name, paths["json_folder"], paths["bins_folder"], ips, ports, sery, params["max_seq_len"], attr_params)
    bins = json.loads(bins_file.read_text())
    if list(bins) != fields:
        raise ValueError("Bins JSON field order differs from model order")
    metadata = dict(field_names=fields, class_mapping=labels,
                    bins_sha256=hashlib.sha256(bins_file.read_bytes()).hexdigest(),
                    input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    seed=seed)
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    data = CustomDataset(str(Path(paths["json_folder"]) / dataset_name) + "/", str(bins_file), labels,
                         params["max_seq_len"], ports + ips, sery, preencode=True)
    loader = DataLoader(data, batch_size=params["batch_size"], shuffle=True)
    model = FlatTransGenerator(len(labels), len(fields), params["max_seq_len"],
                               [len(bins[f]["intervals"]) for f in fields], **config.get("flat_model", {})).to(device)
    print(f"Flat model: packets={params['max_seq_len']}, tokens={model.max_tokens}, device={device}", flush=True)
    train_flat(model, loader, params["epoch"], device, model_dir, params["checkpoint"], metadata)
    del model, loader, data
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    best, metrics = select_flat_checkpoint(config, labels, device, metadata)
    counts = generate_flat_data(config, labels, best, device, metadata)
    result = dict(best_epoch=best, metrics=metrics, generated_counts=counts)
    (model_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as file:
        run_flat(json.load(file))
