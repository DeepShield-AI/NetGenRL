"""Convert UTMobileNet2021 Wild CSV packets to model-shaped flows.

The source has no flow ID or complete TCP flags. The output is a reconstruction,
with its assumptions and losses recorded in processing_report.json.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
import ipaddress
import json
import logging
import math
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo


LOGGER = logging.getLogger("utmn_process")
TIME_RE = re.compile(r"^([A-Z][a-z]{2} \d{1,2}, \d{4} \d\d:\d\d:\d\d)\.(\d{1,9}) (CDT|CST)$")
FILE_RE = re.compile(r"^(.+?)_(?:manual|man-[^_]+)_.*\.csv$", re.IGNORECASE)
META_FIELDS = ("src_ip", "src_port", "dst_ip", "dst_port", "protocol")
PACKET_FIELDS = ("time", "pkt_len", "flags", "ttl")
REQUIRED_CSV_FIELDS = frozenset(("frame.time", "frame.len", "ip.src", "ip.dst", "ip.proto", "ip.ttl",
                                 "tcp.srcport", "tcp.dstport", "tcp.flags.ns", "tcp.flags.fin",
                                 "udp.srcport", "udp.dstport"))


class BadPacket(ValueError):
    pass


def integer(raw, field, low, high):
    try:
        value = Decimal(raw)
        if not value.is_finite() or value != value.to_integral_value():
            raise InvalidOperation
        result = int(value)
    except (TypeError, ValueError, InvalidOperation):
        raise BadPacket("invalid_" + field) from None
    if not low <= result <= high:
        raise BadPacket("out_of_range_" + field)
    return result


def timestamp_ns(raw):
    match = TIME_RE.fullmatch(raw or "")
    if not match:
        raise BadPacket("invalid_frame.time")
    wall, fraction, abbreviation = match.groups()
    local = datetime.strptime(wall, "%b %d, %Y %H:%M:%S").replace(tzinfo=ZoneInfo("America/Chicago"))
    if local.tzname() != abbreviation:
        raise BadPacket("inconsistent_frame.time_zone")
    seconds = int(local.timestamp())
    return seconds * 1_000_000_000 + int(fraction.ljust(9, "0"))


def parse_packet(row, file_name, line_number, stats):
    ethertype = row.get("sll.etype")
    if ethertype == "0x000086dd":
        raise BadPacket("unsupported_ipv6")
    if ethertype not in ("0x00000800", "", None):
        raise BadPacket("unsupported_ethertype")
    try:
        src = ipaddress.IPv4Address(row.get("ip.src", ""))
        dst = ipaddress.IPv4Address(row.get("ip.dst", ""))
    except (ipaddress.AddressValueError, ipaddress.NetmaskValueError, ValueError, TypeError):
        raise BadPacket("invalid_ipv4") from None
    protocol = integer(row.get("ip.proto"), "ip.proto", 0, 255)
    if protocol not in (6, 17):
        raise BadPacket("unsupported_l4")
    prefix = "tcp" if protocol == 6 else "udp"
    sport = integer(row.get(prefix + ".srcport"), prefix + ".srcport", 0, 65535)
    dport = integer(row.get(prefix + ".dstport"), prefix + ".dstport", 0, 65535)
    time_ns = timestamp_ns(row.get("frame.time"))
    length = integer(row.get("frame.len"), "frame.len", 1, 2**32 - 1)
    ttl_raw = row.get("ip.ttl")
    if ttl_raw in ("", None):
        ttl = 0
        stats["ttl_missing_filled_zero"] += 1
    else:
        ttl = integer(ttl_raw, "ip.ttl", 0, 255)
    if protocol == 6:
        flags = 0
        for field, bit in (("tcp.flags.ns", 256), ("tcp.flags.fin", 1)):
            raw = row.get(field)
            if raw in ("", None):
                stats["tcp_flag_cells_missing_filled_zero"] += 1
                continue
            flags |= integer(raw, field, 0, 1) * bit
        # Existing config has flags max 255; nonzero NS cannot be represented.
        if flags > 255:
            raise BadPacket("tcp_ns_outside_model_range")
        stats["tcp_packets_partial_flags"] += 1
    else:
        flags = 0  # TCP flags do not apply to UDP, as in pcap_process.py.
        stats["udp_packets_flags_not_applicable"] += 1
    return (time_ns, line_number, int(src), sport, int(dst), dport, protocol, length, flags, ttl, file_name)


def canonical_key(packet):
    _, _, src, sport, dst, dport, proto, *_ = packet
    return (proto, tuple(sorted(((src, sport), (dst, dport)))))


def create_flow(packets, maximum, stats):
    first = packets[0]
    _, _, src, sport, dst, dport, proto, *_ = first
    series = []
    previous_ns = None
    for packet in packets[:maximum]:
        time_ns, _, psrc, psport, pdst, pdport, _, length, flags, ttl, _ = packet
        if length > 1500:
            stats["retained_packet_lengths_over_config_max_1500"] += 1
        stats["max_retained_frame_len"] = max(stats["max_retained_frame_len"], length)
        delta = 0 if previous_ns is None else round((time_ns - previous_ns) / 1_000_000_000, 2)
        if delta < 0:
            raise ValueError("negative_iat_after_sort")
        direction = 1 if (psrc, psport, pdst, pdport) == (src, sport, dst, dport) else -1
        series.append({"time": delta, "pkt_len": direction * length, "flags": flags, "ttl": ttl})
        previous_ns = time_ns
    stats["output_packets"] += len(series)
    stats["truncated_packets"] += len(packets) - len(series)
    return {"last_time": packets[len(series) - 1][0] / 1_000_000_000, "series": series,
            "src_ip": src, "src_port": sport, "dst_ip": dst,
            "dst_port": dport, "protocol": proto}


def process_file(path, maximum, timeout_ns, stats):
    """One source file at a time; each five-tuple is sorted before timeout split."""
    groups = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("missing_csv_header")
        missing = REQUIRED_CSV_FIELDS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"missing required CSV fields in {path}: {sorted(missing)}")
        for line_number, row in enumerate(reader, 2):
            stats["raw_packets"] += 1
            if None in row or any(value is None for value in row.values()):
                stats["dropped_extra_or_short_csv_row"] += 1
                continue
            try:
                packet = parse_packet(row, path.name, line_number, stats)
            except BadPacket as exc:
                stats["dropped_" + str(exc)] += 1
                continue
            groups[canonical_key(packet)].append(packet)
            stats["accepted_packets"] += 1
    stats["five_tuple_groups"] += len(groups)
    flows = []
    for packets in groups.values():
        # Stable sort retains original CSV order for exactly equal timestamps.
        original = packets
        stats["row_order_time_regressions"] += sum(original[i][0] < original[i - 1][0] for i in range(1, len(original)))
        packets = sorted(packets, key=lambda packet: (packet[0], packet[1]))
        start = 0
        for index in range(1, len(packets) + 1):
            boundary = index == len(packets) or packets[index][0] - packets[index - 1][0] > timeout_ns
            if boundary:
                segment = packets[start:index]
                flows.append(create_flow(segment, maximum, stats))
                if start:
                    stats["idle_timeout_splits"] += 1
                start = index
    return flows


def add_stats(target, source):
    for key, value in source.items():
        if key == "max_retained_frame_len":
            target[key] = max(target[key], value)
        else:
            target[key] += value


def validate_file(path, maximum):
    with path.open(encoding="utf-8") as stream:
        flows = json.load(stream)
    if not isinstance(flows, list):
        raise ValueError("JSON root is not a list")
    packet_count = 0
    for flow in flows:
        if not isinstance(flow, dict) or any(type(flow.get(k)) is not int for k in META_FIELDS):
            raise ValueError("invalid flow metadata")
        if flow["protocol"] not in (6, 17) or not isinstance(flow.get("series"), list) or not 1 <= len(flow["series"]) <= maximum:
            raise ValueError("invalid flow structure or length")
        if not isinstance(flow.get("last_time"), (int, float)) or not math.isfinite(flow["last_time"]):
            raise ValueError("invalid last_time")
        packet_count += len(flow["series"])
        for packet in flow["series"]:
            if not isinstance(packet, dict) or set(packet) != set(PACKET_FIELDS):
                raise ValueError("invalid packet fields")
            if any(type(packet[k]) not in (int, float) or not math.isfinite(packet[k]) for k in PACKET_FIELDS):
                raise ValueError("invalid packet type")
            if packet["time"] < 0 or packet["pkt_len"] == 0 or not 0 <= packet["flags"] <= 255 or not 0 <= packet["ttl"] <= 255:
                raise ValueError("invalid packet value")
    return len(flows), packet_count


def setup_logging(path):
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def validate_loader(output_dir, bins_file, apps, expected):
    if not bins_file.is_file():
        return {"status": "not_run", "reason": "bins_file_missing", "bins_file": str(bins_file)}
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from stateFormer.dataset import CustomDataset

    dataset = CustomDataset(str(output_dir) + "/", str(bins_file),
                            {app: index for index, app in enumerate(apps)}, 16,
                            ["src_port", "dst_port", "src_ip", "dst_ip"],
                            ["time", "pkt_len", "flags", "ttl"])
    if len(dataset) != expected:
        raise ValueError("CustomDataset length mismatch")
    for index in range(len(dataset)):
        sequence, _, _, _ = dataset[index]
        if tuple(sequence.shape) != (16, 8):
            raise ValueError("CustomDataset sequence shape mismatch")
    return {"status": "passed", "bins_file": str(bins_file), "samples_checked": len(dataset),
            "sequence_shape": [16, 8], "caveat": "existing ISCX bins test loader mechanics only; no UTMN bin-fit or TCP semantic validation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", help="Repeat for each source directory")
    parser.add_argument("--output-dir", type=Path, default=Path("/data/xiaozj/utmn"))
    parser.add_argument("--artifacts-dir", type=Path,
                        help="Directory outside the training JSON folder for processing.log and processing_report.json")
    parser.add_argument("--max-packets-per-flow", type=int, default=300)
    parser.add_argument("--idle-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--bins-file", type=Path, default=Path(__file__).resolve().parents[1] / "bins/bins_iscx_small.json")
    args = parser.parse_args()
    if args.max_packets_per_flow < 1 or not math.isfinite(args.idle_timeout_seconds) or args.idle_timeout_seconds <= 0:
        parser.error("packet cap and idle timeout must be positive")
    artifacts_dir = args.artifacts_dir or args.output_dir.with_name(args.output_dir.name + "-artifacts")
    if artifacts_dir.resolve() == args.output_dir.resolve() or args.output_dir.resolve() in artifacts_dir.resolve().parents:
        parser.error("artifacts directory must be outside the training JSON directory")
    input_dirs = args.input_dir or [Path("/data/xiaozj/UTMobileNet2021/Wild Test Data")]
    if len({path.resolve() for path in input_dirs}) != len(input_dirs):
        parser.error("duplicate input directories")
    if len({path.name for path in input_dirs}) != len(input_dirs):
        parser.error("input directory names must be unique for source reporting")
    by_app = defaultdict(list)
    source_files = {}
    for directory in input_dirs:
        if not directory.is_dir():
            parser.error("missing input directory: " + str(directory))
        files = sorted(path for path in directory.rglob("*.csv") if path.is_file())
        if not files:
            parser.error("no CSV files found in " + str(directory))
        source_files[directory.name] = files
        for path in files:
            match = FILE_RE.fullmatch(path.name)
            if not match:
                parser.error("cannot infer application from " + path.name)
            by_app[match.group(1)].append((directory.name, path))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(artifacts_dir / "processing.log")
    LOGGER.info("Inputs %s; output %s; %d CSV; max packets %d; idle timeout %.3f s", [str(p) for p in input_dirs], args.output_dir, sum(map(len, source_files.values())), args.max_packets_per_flow, args.idle_timeout_seconds)
    report = {"input_dirs": [str(p) for p in input_dirs], "output_dir": str(args.output_dir),
              "artifacts_dir": str(artifacts_dir),
              "max_packets_per_flow": args.max_packets_per_flow, "idle_timeout_seconds": args.idle_timeout_seconds,
              "flow_definition": "reconstructed: within each source CSV, bidirectional IPv4 five-tuple, timestamp sort, split after idle timeout; not source-defined true flows",
              "label_source": "CSV filename prefix before _manual_ or _man-[action]_; packet-level application/activity labels absent",
              "flags_limitation": "TCP only NS and FIN observed; absent flags bits written as 0, so TCP flags values are partial and not semantically equivalent to full Scapy TCP flags; UDP 0 means not applicable",
              "frame_length_limitation": "frame.len used as captured original frame length; Scapy len(packet) link-layer equivalence unverified",
              "binning_limitation": "Current config pkt_len max is 1500; longer retained frame.len values are preserved in JSON but existing bins clamp them at their boundary",
              "ipv6_limitation": "CSV has no IPv6 source/destination address columns or hop-limit column; tail-32 conversion cannot be performed from source content, so IPv6 rows are excluded and counted",
              "timezone": "CDT/CST parsed using America/Chicago", "sources": {}, "classes": {}, "totals": {}, "validation": {}}
    total = Counter()
    source_totals = defaultdict(Counter)
    source_apps = defaultdict(lambda: defaultdict(Counter))
    for app, app_files in sorted(by_app.items()):
        stats = Counter()
        flows = []
        for source_name, file in app_files:
            file_stats = Counter()
            file_flows = process_file(file, args.max_packets_per_flow, int(args.idle_timeout_seconds * 1_000_000_000), file_stats)
            file_stats["csv_files"] = 1
            file_stats["reconstructed_flows"] = len(file_flows)
            file_stats["abnormal_records"] = file_stats["raw_packets"] - file_stats["accepted_packets"]
            flows.extend(file_flows)
            add_stats(stats, file_stats)
            add_stats(source_totals[source_name], file_stats)
            add_stats(source_apps[source_name][app], file_stats)
            LOGGER.info("%s/%s: rows=%d accepted=%d flows=%d excluded=%d", source_name, file.name,
                        file_stats["raw_packets"], file_stats["accepted_packets"], file_stats["reconstructed_flows"], file_stats["abnormal_records"])
        output = args.output_dir / (app + ".json")
        with output.open("w", encoding="utf-8") as stream:
            json.dump(flows, stream, ensure_ascii=False, separators=(",", ":"))
        verified_flows, verified_packets = validate_file(output, args.max_packets_per_flow)
        if (verified_flows, verified_packets) != (len(flows), stats["output_packets"]):
            raise ValueError("output verification count mismatch: " + app)
        if stats["reconstructed_flows"] != verified_flows:
            raise ValueError("source flow count mismatch: " + app)
        stats["output_bytes"] = output.stat().st_size
        data = dict(stats)
        data["output_mib"] = round(stats["output_bytes"] / 1024**2, 3)
        data["source_files"] = [source + "/" + path.name for source, path in app_files]
        data["sources"] = {source: dict(source_apps[source][app]) for source in sorted(source_apps) if app in source_apps[source]}
        report["classes"][app] = data
        add_stats(total, stats)
        LOGGER.info("%s: rows=%d accepted=%d reconstructed_flows=%d abnormal=%d size=%d bytes", app, stats["raw_packets"], stats["accepted_packets"], verified_flows, stats["abnormal_records"], stats["output_bytes"])
    for directory in input_dirs:
        name = directory.name
        report["sources"][name] = {"input_dir": str(directory), "input_bytes": sum(path.stat().st_size for path in source_files[name]),
                                   "stats": dict(source_totals[name]),
                                   "applications": {app: dict(stats) for app, stats in sorted(source_apps[name].items())}}
        source_stats = source_totals[name]
        LOGGER.info("SOURCE %s: CSV=%d raw_packets=%d accepted=%d excluded=%d reconstructed_flows=%d output_packets=%d",
                    name, source_stats["csv_files"], source_stats["raw_packets"], source_stats["accepted_packets"],
                    source_stats["abnormal_records"], source_stats["reconstructed_flows"], source_stats["output_packets"])
    report["totals"] = dict(total)
    for key in ("raw_packets", "accepted_packets", "abnormal_records", "reconstructed_flows", "output_packets", "csv_files"):
        if total[key] != sum(source_totals[source][key] for source in source_totals):
            raise ValueError("source total mismatch: " + key)
    actual_outputs = {path.stem for path in args.output_dir.glob("*.json")}
    if actual_outputs != set(by_app):
        raise ValueError(f"unexpected output class JSON files: {sorted(actual_outputs.symmetric_difference(by_app))}")
    report["validation"] = {"json_parse_schema_types_lengths": "passed_all_classes", "files_checked": len(by_app),
                            "loader": validate_loader(args.output_dir, args.bins_file, sorted(by_app), total["reconstructed_flows"])}
    with (artifacts_dir / "processing_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    LOGGER.info("Done: %d source packets; %d accepted; %d reconstructed flows", total["raw_packets"], total["accepted_packets"], total["reconstructed_flows"])


if __name__ == "__main__":
    main()
