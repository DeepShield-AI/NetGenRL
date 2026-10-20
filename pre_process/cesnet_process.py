"""Stream CESNET-TLS22 CSV.gz flows into the project's category JSON schema.

The published PPI contains IPT (ms), direction and transport payload size only.
Endpoint, TTL and per-packet TCP flags are unavailable and are explicit zeros.
"""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import io
import json
import logging
import math
from pathlib import Path
import random
import sys


LOG = logging.getLogger("cesnet_process")
EXPECTED = ("ID", "BYTES", "BYTES_REV", "PACKETS", "PACKETS_REV", "DURATION", "PPI", "PPI_LEN", "PPI_DURATION", "PPI_ROUNDTRIPS", "APP", "CATEGORY")
META = ("src_ip", "src_port", "dst_ip", "dst_port", "protocol")
PACKET = ("time", "pkt_len", "flags", "ttl")


def filename(label):
    # CustomDataset uses filename.split('.')[0] as its class name.
    return label + ".json"


def parse_flow(line, label, max_packets):
    try:
        row = next(csv.reader(io.StringIO(line.decode("utf-8"), newline="")))
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("invalid_csv") from exc
    if len(row) != 30 or row[11] != label:
        raise ValueError("invalid_columns_or_category")
    try:
        ppi = json.loads(row[6])
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid_ppi_json") from exc
    if not isinstance(ppi, list) or len(ppi) != 3 or any(not isinstance(x, list) for x in ppi):
        raise ValueError("invalid_ppi_shape")
    n = len(ppi[0])
    if not n or len(ppi[1]) != n or len(ppi[2]) != n:
        raise ValueError("ppi_length_mismatch_or_empty")
    try:
        if int(row[7]) != n:
            raise ValueError("ppi_len_field_mismatch")
    except (TypeError, ValueError) as exc:
        if str(exc) == "ppi_len_field_mismatch":
            raise
        raise ValueError("invalid_ppi_len") from exc
    series = []
    elapsed = 0.0
    for i, (ipt, direction, size) in enumerate(zip(*ppi)):
        if isinstance(ipt, bool) or not isinstance(ipt, (int, float)) or not math.isfinite(ipt) or ipt < 0:
            raise ValueError("invalid_ipt")
        if direction not in (-1, 1) or isinstance(direction, bool):
            raise ValueError("invalid_direction")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("invalid_size")
        elapsed += ipt / 1000.0
        if i < max_packets:
            series.append({"time": round(ipt / 1000.0, 2) if i else 0.0,
                           "pkt_len": size * direction, "flags": 0, "ttl": 0})
    return {"src_ip": 0, "src_port": 0, "dst_ip": 0, "dst_port": 0,
            "protocol": 6, "last_time": elapsed, "series": series}


def scan(files, limit, seed, max_records=None):
    rng = random.Random(seed)
    raw = Counter()
    reservoir = defaultdict(list)
    failures = Counter()
    file_counts = {}
    total = 0
    for path in files:
        count = 0
        LOG.info("Scanning %s", path)
        try:
            with gzip.open(path, "rb") as source:
                header = source.readline().decode("utf-8-sig").strip().split(",")
                if tuple(header[:12]) != EXPECTED or len(header) != 30:
                    raise ValueError("unexpected CSV header")
                for line in source:
                    total += 1
                    count += 1
                    # CATEGORY precedes exactly 18 trailing numeric columns. PPI's
                    # quoted commas occur only earlier in the line.
                    tail = line.rstrip(b"\r\n").rsplit(b",", 19)
                    if len(tail) != 20:
                        failures["invalid_csv_suffix"] += 1
                        continue
                    try:
                        category = tail[1].decode("utf-8")
                    except UnicodeError:
                        failures["invalid_category_encoding"] += 1
                        continue
                    if not category or "/" in category or "\\" in category:
                        failures["missing_or_unsafe_category"] += 1
                        continue
                    raw[category] += 1
                    selected = reservoir[category]
                    if len(selected) < limit:
                        selected.append(line)
                    else:
                        index = rng.randrange(raw[category])
                        if index < limit:
                            selected[index] = line
                    if max_records is not None and total >= max_records:
                        break
        except (OSError, EOFError, ValueError) as exc:
            LOG.error("Source read failure %s: %s", path, exc)
            failures["source_file_failure"] += 1
            raise
        file_counts[str(path)] = count
        LOG.info("Scanned %s rows from %s", count, path)
        if max_records is not None and total >= max_records:
            break
    return raw, reservoir, failures, file_counts, total


def validate_file(path, max_flows, max_packets):
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list) or len(data) > max_flows:
        raise ValueError(f"invalid flow list: {path}")
    packets = 0
    for item in data:
        if not isinstance(item, dict) or any(type(item.get(key)) is not int for key in META):
            raise ValueError(f"invalid metadata: {path}")
        if not isinstance(item.get("last_time"), (int, float)) or not math.isfinite(item["last_time"]):
            raise ValueError(f"invalid last_time: {path}")
        series = item.get("series")
        if not isinstance(series, list) or not 0 < len(series) <= max_packets:
            raise ValueError(f"invalid series length: {path}")
        for packet in series:
            if not isinstance(packet, dict) or any(not isinstance(packet.get(k), (int, float)) or isinstance(packet[k], bool) or not math.isfinite(packet[k]) for k in PACKET):
                raise ValueError(f"invalid packet: {path}")
        packets += len(series)
    return len(data), packets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/data/xiaozj/cesnet-tls22"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/xiaozj/cesnet-json"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("/data/xiaozj/cesnet-processing-artifacts"))
    parser.add_argument("--max-flows-per-class", type=int, default=2000)
    parser.add_argument("--max-packets-per-flow", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, help="Smoke test only; scan at most this many source records")
    args = parser.parse_args()
    if args.max_flows_per_class < 1 or args.max_packets_per_flow < 1 or (args.max_records is not None and args.max_records < 1):
        parser.error("limits must be positive")
    if args.output_dir.resolve() == args.artifacts_dir.resolve():
        parser.error("artifacts directory must differ from training data directory")
    files = sorted(args.input_dir.rglob("*.csv.gz"))
    if not files:
        parser.error("no source CSV.gz files")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(args.artifacts_dir / "processing.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        LOG.addHandler(handler)
    LOG.info("Arguments: %s", vars(args))
    raw, selected, scan_failures, file_counts, total = scan(files, args.max_flows_per_class, args.seed, args.max_records)
    report = {"input_dir": str(args.input_dir), "output_dir": str(args.output_dir),
              "settings": {"max_flows_per_class": args.max_flows_per_class,
                           "max_packets_per_flow": args.max_packets_per_flow, "seed": args.seed,
                           "max_records": args.max_records}, "source_files": file_counts,
              "raw_rows": total, "scan_failures": dict(scan_failures), "classes": {},
              "limitations": ["PPI IPT is milliseconds; time is seconds rounded to 0.01 as in pcap_process.py.",
                              "PPI SIZE is transport payload bytes, not Scapy packet length.",
                              "IP addresses, ports, TTL and per-packet TCP flags are absent; schema-required values are 0.",
                              "protocol=6 is a dataset-level TLS/TCP assumption, not a source row field.",
                              "last_time is cumulative PPI IPT, not an absolute timestamp.",
                              "Full PPI validation covers selected rows only; unselected PPI errors are unknown."]}
    names = set()
    for category in sorted(raw):
        name = filename(category)
        if name in names:
            raise ValueError("category filename collision")
        names.add(name)
        flows = []
        reasons = Counter()
        for line in selected[category]:
            try:
                flows.append(parse_flow(line, category, args.max_packets_per_flow))
            except ValueError as exc:
                reasons[str(exc)] += 1
        path = args.output_dir / name
        with path.open("w", encoding="utf-8") as stream:
            json.dump(flows, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        n, packets = validate_file(path, args.max_flows_per_class, args.max_packets_per_flow)
        if n != len(flows):
            raise ValueError("output count mismatch")
        size = path.stat().st_size
        report["classes"][category] = {"filename": name, "raw_flows": raw[category],
            "sampled_flows": len(selected[category]), "successful_flows": n,
            "failed_flows": sum(reasons.values()), "failure_reasons": dict(reasons),
            "packets": packets, "size_bytes": size, "size_mib": round(size / 1048576, 3)}
        LOG.info("%s: raw=%s sampled=%s success=%s failed=%s packets=%s bytes=%s", category, raw[category], len(selected[category]), n, sum(reasons.values()), packets, size)
    report["totals"] = {key: sum(c[key] for c in report["classes"].values()) for key in ("raw_flows", "sampled_flows", "successful_flows", "failed_flows", "packets", "size_bytes")}
    report["totals"]["source_excluded_rows"] = sum(scan_failures.values())
    if report["totals"]["raw_flows"] + report["totals"]["source_excluded_rows"] != total:
        raise ValueError("source row accounting mismatch")
    with (args.artifacts_dir / "processing_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    LOG.info("Completed: %s", report["totals"])


if __name__ == "__main__":
    main()
