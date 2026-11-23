"""Flat-model selection/generation, reusing the existing metrics and decoding."""
from pathlib import Path
import json
import math

import numpy as np
import ot

from post_process.check_bestmodel import get_real_data, get_fake_data, compute_cost_matrices, pad
from post_process.data_save import get_gen_data, save_data
from stateFormer.flat_train import load_flat_checkpoint


def select_flat_checkpoint(config, labels, device, metadata):
    paths, params, attrs = config["path"], config["model_paras"], config["attributes"]
    dataset = paths["dataset"]
    sery = list(attrs["sery_attribute"])
    meta = list(attrs["port_attribute"]) + list(attrs["ip_attribute"])
    bins = json.loads((Path(paths["bins_folder"]) / f"bins_{dataset}.json").read_text())
    real = get_real_data(str(Path(paths["json_folder"]) / dataset) + "/", labels, meta, sery, bins, params["max_seq_len"])
    best, minimum, metrics = None, math.inf, []
    for epoch in range(params["checkpoint"], params["epoch"] + 1, params["checkpoint"]):
        path = Path(paths["model_folder"]) / dataset / f"generator_{epoch}.pth"
        generator, _ = load_flat_checkpoint(path, device, metadata)
        generator.eval()
        ot_sum = hm_sum = 0.0
        for label in labels:
            fake = get_fake_data(labels, len(labels), real[label], label, generator, bins, sery, meta, params["batch_size"], device)
            a = np.array([pad(seq, params["max_seq_len"]) for seq in real[label]])
            b = np.array([pad(seq, params["max_seq_len"]) for seq in fake])
            l2, hamming = compute_cost_matrices(a, b, device)
            ot_sum += float(ot.emd2([], [], l2))
            hm_sum += float(ot.emd2([], [], hamming))
        if not math.isfinite(ot_sum) or not math.isfinite(hm_sum):
            raise FloatingPointError("Non-finite checkpoint evaluation")
        metrics.append(dict(epoch=epoch, ot=ot_sum, hm=hm_sum))
        print(f"model {epoch} OT: {ot_sum} HM: {hm_sum}", flush=True)
        if ot_sum < minimum:
            best, minimum = epoch, ot_sum
        del generator
    if best is None:
        raise ValueError("No checkpoint evaluated")
    print(f"Best model is {best}.", flush=True)
    return best, metrics


def generate_flat_data(config, labels, epoch, device, metadata):
    paths, params, attrs = config["path"], config["model_paras"], config["attributes"]
    dataset = paths["dataset"]
    sery = list(attrs["sery_attribute"])
    meta = list(attrs["port_attribute"]) + list(attrs["ip_attribute"])
    bins = json.loads((Path(paths["bins_folder"]) / f"bins_{dataset}.json").read_text())
    real = get_real_data(str(Path(paths["json_folder"]) / dataset) + "/", labels, meta, sery, bins, params["max_seq_len"])
    generator, _ = load_flat_checkpoint(Path(paths["model_folder"]) / dataset / f"generator_{epoch}.pth", device, metadata)
    generator.eval()
    output = Path(paths["result_folder"]) / dataset
    counts = {}
    for label in labels:
        if (output / f"{label}.json").exists():
            raise FileExistsError(f"Refusing to overwrite generated output for {label}")
        generated = []
        for _ in range(params["expand_times"] + 1):
            generated.extend(get_gen_data(labels, len(labels), real[label], label, generator, bins, sery, meta, params["batch_size"], device))
        save_data(label, generated, meta, sery, str(output) + "/")
        counts[label] = len(generated)
    return counts
