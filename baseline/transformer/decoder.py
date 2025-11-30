import struct
from tokenizer import *
import json
# Reverse process: byte sequence -> flow JSON

def decode_flow(byte_seq):
    idx = 2
    flow = {}

    print(byte_seq)
    # ---- 1. decode header ----
    src_ip = struct.unpack(">I", bytes(byte_seq[idx:idx+4]))[0]
    idx += 4
    dst_ip = struct.unpack(">I", bytes(byte_seq[idx:idx+4]))[0]
    idx += 4
    src_port = struct.unpack(">H", bytes(byte_seq[idx:idx+2]))[0]
    idx += 2
    dst_port = struct.unpack(">H", bytes(byte_seq[idx:idx+2]))[0]
    idx += 2
    # protocol = byte_seq[idx]
    # idx += 1


    flow["src_ip"] = src_ip
    flow["dst_ip"] = dst_ip
    flow["src_port"] = src_port
    flow["dst_port"] = dst_port
    # flow["protocol"] = protocol
    flow["series"] = []


    # ---- 2. decode packets ----
    FIELD_SIZES = {
        "time": 8,      # float64
        "pkt_len": 2,   # uint16
        "direction": 1, # 1 byte
        "flags": 1,     # 1 byte
        "ttl": 1        # 1 byte
    }

    while idx < len(byte_seq):
        if byte_seq[idx] == TOKEN_EOS:
            break

        pkt = {}
    
        # --- time ---
        if idx + FIELD_SIZES["time"] > len(byte_seq) or TOKEN_SEP in byte_seq[idx:idx+FIELD_SIZES["time"]]:
            pkt["time"] = 0.0
            # 找到 SEP 的位置并跳过
            sep_pos = idx
            while sep_pos < len(byte_seq) and byte_seq[sep_pos] != TOKEN_SEP:
                sep_pos += 1
            idx = sep_pos + 1 if sep_pos < len(byte_seq) else len(byte_seq)
            pkt["pkt_len"] = 0
            pkt["direction"] = 0
            pkt["flags"] = 0
            pkt["ttl"] = 0
            flow["series"].append(pkt)
            continue
        pkt["time"] = struct.unpack(">d", bytes(byte_seq[idx:idx+8]))[0]
        idx += 8

        # --- pkt_len ---
        if idx + FIELD_SIZES["pkt_len"] > len(byte_seq) or TOKEN_SEP in byte_seq[idx:idx+FIELD_SIZES["pkt_len"]]:
            pkt["pkt_len"] = 0
            pkt["direction"] = 0
            pkt["flags"] = 0
            pkt["ttl"] = 0
            # 跳到 SEP
            sep_pos = idx
            while sep_pos < len(byte_seq) and byte_seq[sep_pos] != TOKEN_SEP:
                sep_pos += 1
            idx = sep_pos + 1 if sep_pos < len(byte_seq) else len(byte_seq)
            flow["series"].append(pkt)
            continue
        pkt_len = struct.unpack(">H", bytes(byte_seq[idx:idx+2]))[0]
        idx += 2

        # --- direction ---
        if idx >= len(byte_seq) or byte_seq[idx] == TOKEN_SEP:
            pkt["pkt_len"] = pkt_len
            pkt["direction"] = 0
            pkt["flags"] = 0
            pkt["ttl"] = 0
            idx += 1 if idx < len(byte_seq) else 0
            flow["series"].append(pkt)
            continue
        direction = byte_seq[idx]
        idx += 1
        if direction == 1:
            pkt_len = -pkt_len
        pkt["pkt_len"] = pkt_len
        pkt["direction"] = direction

        # --- flags ---
        if idx >= len(byte_seq) or byte_seq[idx] == TOKEN_SEP:
            pkt["flags"] = 0
            pkt["ttl"] = 0
            idx += 1 if idx < len(byte_seq) else 0
            flow["series"].append(pkt)
            continue
        pkt["flags"] = byte_seq[idx]
        idx += 1

        # --- ttl ---
        if idx >= len(byte_seq) or byte_seq[idx] == TOKEN_SEP:
            pkt["ttl"] = 0
            idx += 1 if idx < len(byte_seq) else 0
            flow["series"].append(pkt)
            continue
        pkt["ttl"] = byte_seq[idx]
        idx += 1

        # --- PKT_END ---
        if idx < len(byte_seq) and byte_seq[idx] == TOKEN_SEP:
            idx += 1

        print(pkt)
        flow["series"].append(pkt)

    print(flow)
    return flow

def save_flow_json(flow, path):
    with open(path, "w") as f:
        json.dump(flow, f, indent=2)