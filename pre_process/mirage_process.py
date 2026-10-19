"""Stream MIRAGE flow records into the JSON shape consumed by CustomDataset.

The source is read only. Sampling is global per BF_label, independent of the
source directory. See processing_report.json for mappings and limitations.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import ipaddress
import json
import logging
import math
from pathlib import Path
import random
import sys


PACKET_FIELDS = ("timestamp", "src_port", "dst_port", "packet_dir", "IP_packet_bytes", "iat", "TCP_flags")
FLAG_BITS = {"F": 1, "S": 2, "R": 4, "P": 8, "A": 16, "U": 32, "E": 64, "C": 128, "N": 256}
LOGGER = logging.getLogger("mirage_process")


def filename_for_label(label):
    # Dots break CustomDataset's filename.split('.')[0] label lookup.
    raw = label.encode("utf-8")
    return "".join(chr(b) if 48 <= b <= 57 or 65 <= b <= 90 or 97 <= b <= 122 or b in (45, 95) else f"%{b:02X}" for b in raw) + ".json"


def parse_scalar_line(line):
    return json.loads(line.split(b":", 1)[1].strip().rstrip(b","))


def scan_sources(files, maximum, seed):
    """First pass: read lines, retain only offsets for reservoir samples."""
    rng = random.Random(seed)
    raw_counts = Counter()
    source_failures = Counter()
    selected = defaultdict(list)
    file_errors = []
    total = 0
    for file_id, path in enumerate(files):
        try:
            with path.open("rb") as stream:
                active = False
                label = None
                label_seen = False
                key = None
                in_packet_data = False
                current_array = None
                array_offsets = {}
                array_lengths = Counter()
                while True:
                    position = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if line.startswith(b'    "') and line.rstrip().endswith(b": {"):
                        if active:
                            source_failures["unfinished_flow"] += 1
                        active = True
                        try:
                            key = json.loads(line.strip()[:-3])
                        except (ValueError, UnicodeDecodeError):
                            key = None
                        label = None
                        label_seen = False
                        in_packet_data = False
                        current_array = None
                        array_offsets = {}
                        array_lengths = Counter()
                    elif active and current_array is not None:
                        if line.strip() in (b"]", b"],"):
                            current_array = None
                        else:
                            array_lengths[current_array] += 1
                    elif active and line.startswith(b'        "packet_data": {'):
                        in_packet_data = True
                    elif active and in_packet_data and line.startswith(b'        },'):
                        in_packet_data = False
                    elif active and in_packet_data and line.startswith(b'            "'):
                        name = line.split(b'"', 2)[1].decode("ascii", "ignore")
                        if name in PACKET_FIELDS and line.rstrip().endswith(b"["):
                            current_array = name
                            array_offsets[name] = stream.tell()
                    elif active and line.startswith(b'            "BF_label":'):
                        try:
                            label = parse_scalar_line(line)
                            label_seen = True
                        except (ValueError, UnicodeDecodeError):
                            source_failures["invalid_BF_label"] += 1
                    elif active and line.strip() in (b"},", b"}") and line.startswith(b"    }"):
                        total += 1
                        if not label_seen or not isinstance(label, str) or not label:
                            source_failures["missing_or_invalid_BF_label"] += 1
                            raw_counts["<missing>"] += 1
                        else:
                            raw_counts[label] += 1
                            slot = selected[label]
                            descriptor = (file_id, key, dict(array_offsets), dict(array_lengths))
                            if len(slot) < maximum:
                                slot.append(descriptor)
                            else:
                                index = rng.randrange(raw_counts[label])
                                if index < maximum:
                                    slot[index] = descriptor
                        active = False
                if active:
                    source_failures["unfinished_flow"] += 1
        except OSError as exc:
            file_errors.append({"file": str(path), "error": str(exc)})
        if (file_id + 1) % 100 == 0:
            LOGGER.info("Source scan: %d/%d files; %d flows", file_id + 1, len(files), total)
    return raw_counts, selected, source_failures, file_errors, total


def read_selected(path, key, offsets, lengths, maximum):
    """Seek directly to each selected packet array and read its retained prefix."""
    arrays = {name: [] for name in PACKET_FIELDS}
    with path.open("rb") as stream:
        for name in PACKET_FIELDS:
            if name not in offsets:
                raise ValueError("missing_packet_array_" + name)
            stream.seek(offsets[name])
            for _ in range(min(lengths.get(name, 0), maximum)):
                line = stream.readline()
                if not line or line.strip() in (b"]", b"],"):
                    raise ValueError("truncated_packet_array_" + name)
                arrays[name].append(json.loads(line.strip().rstrip(b",")))
    if key is None:
        raise ValueError("missing_5tuple")
    return key, arrays, lengths


def numeric(value, name, integral=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("invalid_" + name)
    if integral and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError("invalid_" + name)
    return value


def convert_record(key, arrays, lengths, maximum):
    parts = key.split(",")
    if len(parts) != 5:
        raise ValueError("invalid_5tuple")
    try:
        src_ip = ipaddress.IPv4Address(parts[0])
        dst_ip = ipaddress.IPv4Address(parts[2])
        src_port, dst_port, protocol = int(parts[1]), int(parts[3]), int(parts[4])
    except (ValueError, ipaddress.AddressValueError) as exc:
        raise ValueError("invalid_5tuple") from exc
    if not 0 <= src_port <= 65535 or not 0 <= dst_port <= 65535 or protocol not in (6, 17):
        raise ValueError("invalid_5tuple")
    n = lengths["timestamp"]
    if n <= 0 or any(lengths[name] != n for name in PACKET_FIELDS):
        raise ValueError("packet_array_length_mismatch")
    count = min(n, maximum)
    first_dir = arrays["packet_dir"][0]
    if first_dir not in (0, 1):
        raise ValueError("invalid_packet_dir")
    if first_dir == 1:
        src_ip, dst_ip = dst_ip, src_ip
        src_port, dst_port = dst_port, src_port
    series = []
    previous_timestamp = None
    for index in range(count):
        timestamp = numeric(arrays["timestamp"][index], "timestamp")
        iat = numeric(arrays["iat"][index], "iat")
        packet_bytes = numeric(arrays["IP_packet_bytes"][index], "IP_packet_bytes", integral=True)
        direction = arrays["packet_dir"][index]
        sport = numeric(arrays["src_port"][index], "src_port", integral=True)
        dport = numeric(arrays["dst_port"][index], "dst_port", integral=True)
        if packet_bytes <= 0:
            raise ValueError("invalid_IP_packet_bytes")
        if packet_bytes > 1500:
            raise ValueError("IP_packet_bytes_out_of_bin_range")
        if direction not in (0, 1):
            raise ValueError("invalid_packet_dir")
        if (sport, dport) != ((src_port, dst_port) if direction == first_dir else (dst_port, src_port)):
            raise ValueError("packet_port_direction_mismatch")
        delta = 0.0 if index == 0 else timestamp - previous_timestamp
        if delta < -0.000001 or iat < -0.000001 or abs(delta - iat) > 0.01:
            raise ValueError("timestamp_iat_mismatch")
        flags_text = arrays["TCP_flags"][index]
        if not isinstance(flags_text, str):
            raise ValueError("invalid_TCP_flags")
        if protocol == 17:
            if flags_text:
                raise ValueError("UDP_with_nonempty_TCP_flags")
            flags = 0  # The existing PCAP preprocessor uses 0 for UDP.
        else:
            if len(set(flags_text)) != len(flags_text) or any(c not in FLAG_BITS for c in flags_text):
                raise ValueError("invalid_TCP_flags")
            flags = sum(FLAG_BITS[c] for c in flags_text)
        if flags > 255:
            raise ValueError("TCP_flags_out_of_bin_range")
        model_time = 0 if index == 0 else round(delta, 2)
        if model_time > 10000:
            raise ValueError("time_out_of_bin_range")
        series.append({
            "time": model_time,
            "pkt_len": packet_bytes if direction == first_dir else -packet_bytes,
            "flags": flags,
            "ttl": 0,  # Explicit user-requested numeric default; source has no TTL.
        })
        previous_timestamp = timestamp
    return {"last_time": float(previous_timestamp), "series": series,
            "src_ip": int(src_ip), "src_port": src_port,
            "dst_ip": int(dst_ip), "dst_port": dst_port, "protocol": protocol}, n


def validate_class_file(path, max_flows, max_packets, bins_file):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from stateFormer.dataset import CustomDataset

    with path.open(encoding="utf-8") as stream:
        flows = json.load(stream)
    if not isinstance(flows, list) or len(flows) > max_flows:
        raise ValueError("invalid_output_flow_count")
    packets = 0
    for flow in flows:
        if not isinstance(flow, dict) or not flow.get("series"):
            raise ValueError("invalid_output_flow")
        if any(not isinstance(flow.get(k), int) or isinstance(flow[k], bool) for k in ("src_ip", "src_port", "dst_ip", "dst_port", "protocol")):
            raise ValueError("invalid_output_meta_type")
        if len(flow["series"]) > max_packets:
            raise ValueError("invalid_output_packet_count")
        packets += len(flow["series"])
        for packet in flow["series"]:
            if set(packet) != {"time", "pkt_len", "flags", "ttl"}:
                raise ValueError("invalid_output_packet_fields")
            if any(isinstance(packet[k], bool) or not isinstance(packet[k], (int, float)) or not math.isfinite(packet[k]) for k in packet):
                raise ValueError("invalid_output_packet_type")
    count = len(flows)
    del flows
    gc.collect()
    # One class at a time: this invokes the real loader without loading all classes.
    if count:
        dataset = CustomDataset(str(path.parent) + "/", str(bins_file), {path.stem: 0},
                                16, ["src_port", "dst_port", "src_ip", "dst_ip"],
                                ["time", "pkt_len", "flags", "ttl"])
        if len(dataset) != count:
            raise ValueError("loader_length_mismatch")
        for index in (0, count - 1):
            sequence, label, length, weight = dataset[index]
            if tuple(sequence.shape) != (16, 8) or not (sequence[:int(length)] > 0).all():
                raise ValueError("loader_sample_incompatible")
        del dataset
        gc.collect()
    return count, packets


def run(args):
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.is_dir():
        raise ValueError("input_dir_missing")
    if output_dir == input_dir or input_dir in output_dir.parents or output_dir in input_dir.parents:
        raise ValueError("input_output_overlap")
    if args.max_flows_per_class <= 0 or args.max_packets_per_flow <= 0:
        raise ValueError("limits_must_be_positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output_dir_not_empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(output_dir / "processing.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    files = sorted(input_dir.rglob("*.json"))
    LOGGER.info("Input: %s; output: %s; files: %d; seed: %d", input_dir, output_dir, len(files), args.seed)
    raw_counts, selected, source_failures, file_errors, total = scan_sources(files, args.max_flows_per_class, args.seed)
    if file_errors or source_failures.get("unfinished_flow"):
        raise RuntimeError("source_scan_incomplete; inspect processing.log")
    LOGGER.info("Source scan complete: %d flows; %d classes", total, len(selected))
    bins_file = Path(__file__).resolve().parents[1] / "bins" / "bins_iscx_small.json"
    if not bins_file.is_file():
        raise ValueError("compatibility_bins_missing")
    report = {
        "input_dir": str(input_dir), "output_dir": str(output_dir),
        "command_parameters": vars(args), "started_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": len(files), "raw_flows_total": total,
        "source_failures": dict(source_failures), "file_errors": file_errors,
        "classes": {}, "mapping_notes": [
            "IP_packet_bytes is an IP-layer length. Original pcap_process uses Scapy len(packet); L2 length cannot be recovered from this JSON.",
            "TTL is absent from source and set to numeric 0 by explicit request.",
            "UDP flags are set to 0 as in pcap_process; TCP flag strings are mapped to bit masks.",
            "Labels containing punctuation are percent-encoded in filenames because CustomDataset splits filenames at the first dot; label mapping is recorded per class.",
            "Sampling is seeded global reservoir sampling by flow_metadata.BF_label. A selected invalid flow is counted as failed and is not replaced.",
            "Each successful flow keeps the first max_packets_per_flow packets; time is rounded timestamp difference to two decimal places as in pcap_process.",
        ],
    }
    used_names = set()
    for label in sorted(selected):
        filename = filename_for_label(label)
        if filename in used_names:
            raise ValueError("output_filename_collision")
        used_names.add(filename)
        path = output_dir / filename
        failures = Counter()
        failure_examples = []
        success = 0
        packets = 0
        source_packets = 0
        descriptors = sorted(selected[label])
        with path.open("w", encoding="utf-8") as stream:
            stream.write("[")
            for file_id, key, offsets, source_lengths in descriptors:
                try:
                    key, arrays, lengths = read_selected(files[file_id], key, offsets, source_lengths, args.max_packets_per_flow)
                    record, original_length = convert_record(key, arrays, lengths, args.max_packets_per_flow)
                except (ValueError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    reason = str(exc) if isinstance(exc, ValueError) and str(exc).startswith(("invalid_", "packet_", "timestamp_", "UDP_", "selected_", "missing_", "unfinished_", "unexpected_")) else type(exc).__name__
                    failures[reason] += 1
                    if len(failure_examples) < 5:
                        failure_examples.append({"file": str(files[file_id]), "offset": min(offsets.values(), default=0), "reason": reason})
                    continue
                if success:
                    stream.write(",")
                json.dump(record, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                success += 1
                packets += len(record["series"])
                source_packets += original_length
            stream.write("]\n")
        validated_flows, validated_packets = validate_class_file(path, args.max_flows_per_class, args.max_packets_per_flow, bins_file)
        if (validated_flows, validated_packets) != (success, packets):
            raise ValueError("postwrite_validation_mismatch")
        size = path.stat().st_size
        report["classes"][label] = {
            "output_file": filename, "raw_flows": raw_counts[label],
            "sampled_flows": len(descriptors), "successful_flows": success,
            "failed_flows": sum(failures.values()), "skipped_by_sampling": raw_counts[label] - len(descriptors),
            "failure_reasons": dict(failures), "failure_examples": failure_examples,
            "source_packets_in_successful_flows": source_packets,
            "packets_written": packets, "file_bytes": size, "file_mib": round(size / 2**20, 3),
            "validation": "passed",
        }
        LOGGER.info("Class %s: raw=%d sampled=%d success=%d failed=%d packets=%d bytes=%d (%.3f MiB)",
                    label, raw_counts[label], len(descriptors), success, sum(failures.values()), packets, size, size / 2**20)
    if raw_counts.get("<missing>"):
        report["classes"]["<missing>"] = {
            "output_file": None, "raw_flows": raw_counts["<missing>"], "sampled_flows": 0,
            "successful_flows": 0, "failed_flows": raw_counts["<missing>"],
            "skipped_by_sampling": 0, "failure_reasons": {"missing_or_invalid_BF_label": raw_counts["<missing>"]},
            "packets_written": 0, "file_bytes": 0, "file_mib": 0, "validation": "not_applicable",
        }
    report["totals"] = {
        "raw_flows": sum(c["raw_flows"] for c in report["classes"].values()),
        "sampled_flows": sum(c["sampled_flows"] for c in report["classes"].values()),
        "successful_flows": sum(c["successful_flows"] for c in report["classes"].values()),
        "failed_flows": sum(c["failed_flows"] for c in report["classes"].values()),
        "packets_written": sum(c["packets_written"] for c in report["classes"].values()),
        "file_bytes": sum(c["file_bytes"] for c in report["classes"].values()),
    }
    if report["totals"]["raw_flows"] != total:
        raise ValueError("raw_count_reconciliation_failed")
    report["validation"] = "passed"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    with (output_dir / "processing_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    LOGGER.info("Complete: %s", report["totals"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="/data/xiaozj/MIRAGE-AppAct-2024")
    parser.add_argument("--output-dir", default="/data/xiaozj/mirage-json")
    parser.add_argument("--max-flows-per-class", type=int, default=2000)
    parser.add_argument("--max-packets-per-flow", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20241016)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
