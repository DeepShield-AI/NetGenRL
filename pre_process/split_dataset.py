"""Split each class JSON under data-json into reproducible 4:1 train/test files.

Usage: python pre_process/split_dataset.py --input-dir data-json/mirage --seed 42
The source directory is read only. Existing output directories are never replaced.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile


DATA_ROOT = Path(__file__).resolve().parents[1] / "data-json"
META_FIELDS = ("src_ip", "src_port", "dst_ip", "dst_port", "protocol")
SERIES_FIELDS = ("time", "pkt_len", "flags", "ttl")


def validate_flows(flows, path):
    if not isinstance(flows, list):
        raise ValueError(f"{path}: top-level JSON must be a flow list")
    if not flows:
        raise ValueError(f"{path}: empty class cannot be loaded by CustomDataset")
    for index, flow in enumerate(flows):
        if not isinstance(flow, dict) or any(type(flow.get(key)) is not int for key in META_FIELDS):
            raise ValueError(f"{path}: flow {index} has invalid metadata")
        series = flow.get("series")
        if not isinstance(series, list) or not series:
            raise ValueError(f"{path}: flow {index} has no packet series")
        for packet_index, packet in enumerate(series):
            if not isinstance(packet, dict) or any(
                key not in packet or type(packet[key]) not in (int, float)
                or not math.isfinite(packet[key]) for key in SERIES_FIELDS
            ):
                raise ValueError(f"{path}: flow {index} packet {packet_index} has invalid model fields")


def split_indices(count, filename, seed):
    if count == 1:
        return set(range(count)), set()
    target_test = max(1, min(count - 1, round(count / 5)))
    class_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{filename}".encode("utf-8")).digest(), "big"
    )
    rng = random.Random(class_seed)
    indices = list(range(count))
    rng.shuffle(indices)
    test_indices = set(indices[:target_test])
    train_indices = set(range(count)) - test_indices
    assert train_indices and train_indices.isdisjoint(test_indices)
    assert len(train_indices) + len(test_indices) == count
    return train_indices, test_indices


def split_file(source, train_path, test_path, seed):
    with source.open("r", encoding="utf-8") as stream:
        flows = json.load(stream)
    validate_flows(flows, source)
    train_indices, test_indices = split_indices(len(flows), source.name, seed)
    train = [flow for index, flow in enumerate(flows) if index in train_indices]
    test = [flow for index, flow in enumerate(flows) if index in test_indices]
    for path, subset in ((train_path, train), (test_path, test)):
        with path.open("w", encoding="utf-8") as stream:
            json.dump(subset, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # Reparse the written files and verify exact membership by source index.
    with train_path.open(encoding="utf-8") as stream:
        written_train = json.load(stream)
    with test_path.open(encoding="utf-8") as stream:
        written_test = json.load(stream)
    if written_train != train or written_test != test:
        raise ValueError(f"{source}: written JSON differs from selected source flows")
    if train_indices & test_indices or train_indices | test_indices != set(range(len(flows))):
        raise ValueError(f"{source}: repeated or missing source index")
    if len(written_train) + len(written_test) != len(flows):
        raise ValueError(f"{source}: missing or repeated source flow")
    return len(flows), len(train), len(test)


def split_dataset(input_dir, seed):
    root = DATA_ROOT.resolve()
    source = input_dir.resolve()
    if source.parent != root or not source.is_dir():
        raise ValueError(f"input must be an existing direct child of {root}")
    files = sorted(source.glob("*.json"))
    if not files or any(not path.is_file() or path.suffix != ".json" for path in source.iterdir()):
        raise ValueError("input must contain only class JSON files")
    train_dir = root / f"{source.name}-train"
    test_dir = root / f"{source.name}-test"
    if train_dir.exists() or test_dir.exists():
        raise FileExistsError("output directory already exists; refusing to overwrite it")

    staging = Path(tempfile.mkdtemp(prefix=f".{source.name}-split-", dir=root))
    staged_train = staging / train_dir.name
    staged_test = staging / test_dir.name
    staged_train.mkdir()
    staged_test.mkdir()
    results = []
    try:
        for file in files:
            counts = split_file(file, staged_train / file.name, staged_test / file.name, seed)
            results.append((file.name, *counts))
        if {path.name for path in staged_train.iterdir()} != {file.name for file in files}:
            raise ValueError("train class filenames do not match source")
        if {path.name for path in staged_test.iterdir()} != {file.name for file in files}:
            raise ValueError("test class filenames do not match source")
        os.replace(staged_train, train_dir)
        try:
            os.replace(staged_test, test_dir)
        except Exception:
            os.replace(train_dir, staged_train)
            raise
    finally:
        shutil.rmtree(staging)
    return train_dir, test_dir, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Direct child directory of data-json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_dir, test_dir, results = split_dataset(args.input_dir, args.seed)
    print("class\toriginal\ttrain\ttest")
    for row in results:
        print("\t".join(map(str, row)))
    print("TOTAL\t" + "\t".join(str(sum(row[column] for row in results)) for column in range(1, 4)))
    print(f"verified: no missing or repeated source indices; output: {train_dir}, {test_dir}")


if __name__ == "__main__":
    main()
