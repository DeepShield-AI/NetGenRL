import json
import struct
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import *

class FlowByteDataset(Dataset):
    """
    将 JSON 流转为： [IP,Port,Protocol... Packet_bytes ...]
    """

    def __init__(self, json_dir, labels, max_len=8192):
        self.samples = []
        for label, id in labels.items():
            path = f"{json_dir}/{label}.json"
            flows = json.load(open(path, "r"))
            for flow in flows:
                seq = self.encode_flow(flow,id)
                if len(seq) <= max_len:
                    self.samples.append(seq)

    def encode_ip(self, ip_int):
        """32-bit IP -> 4 字节"""
        return list(struct.pack(">I", ip_int))

    def encode_port(self, port):
        """port -> 2 字节"""
        port = max(0, min(int(port), 65535))
        return list(struct.pack(">H", port))

    def encode_flow_header(self, flow):
        """
        Flow-level header:
        src_ip(4) dst_ip(4) src_port(2) dst_port(2) protocol(1)
        = 13 bytes
        """
        header = []
        header += self.encode_ip(flow["src_ip"])
        header += self.encode_ip(flow["dst_ip"])
        header += self.encode_port(flow["src_port"])
        header += self.encode_port(flow["dst_port"])
        # proto = max(0, min(int(flow.get("protocol", 0)), 255))
        # header.append(proto)
        return header

    def encode_packet(self, pkt):
        seq = []

        # time: float64 → 8 bytes
        t = float(pkt["time"])
        seq.extend(list(struct.pack(">d", t)))

        # pkt_len: unsigned 2 bytes
        plen = abs(int(pkt["pkt_len"]))
        plen = max(0, min(plen, 65535))
        seq.extend(list(struct.pack(">H", plen)))

        # direction: sign
        direction = 0 if pkt["pkt_len"] > 0 else 1
        seq.append(direction)

        # flags: 1 byte
        flags = max(0, min(int(pkt.get("flags", 0)), 255))
        seq.append(flags)

        # ttl: 1 byte
        ttl = max(0, min(int(pkt.get("ttl", 0)), 255))
        seq.append(ttl)

        # packet-end
        seq.append(TOKEN_SEP)

        return seq

    def encode_flow(self, flow, id):
        """
        Flow bytes:
        [FlowHeader(13 bytes), packet_bytes..., FLOW_END]
        """
        seq = []
        seq.append(id)
        seq.append(TOKEN_SEP)

        # --- Flow Header ---
        seq += self.encode_flow_header(flow)

        # --- Packet Series ---
        for pkt in flow["series"]:
            seq += self.encode_packet(pkt)

        seq.append(TOKEN_EOS)
        return seq

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """
    Return: ids tensor + attn mask
    """
    max_len = max(len(seq) for seq in batch)

    ids = []
    attn = []

    for seq in batch:
        pad_len = max_len - len(seq)
        ids.append(seq + [0] * pad_len)
        attn.append([1] * len(seq) + [0] * pad_len)

    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(attn, dtype=torch.bool)
    )
