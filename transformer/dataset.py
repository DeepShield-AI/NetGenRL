import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformer.config import *
import struct

class FlattenTransformerDataset(Dataset):

    def __init__(self, data_folder, bins_file, class_mapping,
                 max_seq_len, port_attrs, ip_attrs, sery_attrs):

        self.data_folder = data_folder
        self.class_mapping = class_mapping
        self.max_seq_len = max_seq_len   # packet 数量
        self.port_attrs = port_attrs     # 例如：["src_port", "dst_port"]
        self.ip_attrs = ip_attrs         # 例如：["src_ip", "dst_ip"]
        self.sery_attrs = sery_attrs       # 例如：["time","pkt_len","flags","ttl"]

        # 加载全部 JSON 数据
        self.data = []
        self.labels = []
        label_count = {}
        for filename in os.listdir(self.data_folder):
            label_name = filename.split('.')[0]
            if label_name not in class_mapping:
                continue

            with open(self.data_folder + filename, 'r') as f:
                json_data = json.load(f)
                self.data.extend(json_data)
                label_count[label_name] = 1/len(json_data)
                self.labels.extend([label_name] * len(json_data))

        # 分箱区间
        with open(bins_file, 'r') as f:
            self.bins_data = json.load(f)

        # ---------------------------------------------------------
        # 为每个字段分配 vocab offset
        # ---------------------------------------------------------
        # print(self.port_attrs, self.sery_attrs, self.ip_attrs)
        self.field_list = (["label_token"] + ["len_token"] + self.port_attrs + self.sery_attrs + self.ip_attrs + ["begin_token"] + ["pre_token"] + ["sep_token"] + ["suf_token"] + ["end_token"] + ["padding_token"])
        # self.field_list = (["byte_token"] + ["label_token"] + ["len_token"]  + ["begin_token"] + ["pre_token"] + ["sep_token"] + ["suf_token"] + ["end_token"] + ["padding_token"])

        self.field_offsets = {}      # 字段→开始token id
        self.field_vocab_sizes = {}  # 字段→该字段vocab大小

        current_offset = 0
        for field in self.field_list:
            if field == "label_token":
                vocab_size = len(class_mapping)  # label 的离散数量
            elif field == "len_token":
                vocab_size = max_seq_len
            elif field == "sep_token":
                vocab_size = len(SEPS)  # sep 的离散数量
            elif field == "suf_token" or field == "end_token" or field == "padding_token" or field == "begin_token" or field == "pre_token":
                vocab_size = 1
            elif field == "pkt_len":
                vocab_size = 3001
            else:
                # vocab_size = len(self.bins_data[field]["intervals"])
                vocab_size = 256

            self.field_offsets[field] = current_offset
            self.field_vocab_sizes[field] = vocab_size
            current_offset += vocab_size

        self.total_vocab_size = current_offset  # Transformer 统一 vocab_size
        
        self.label_weight = {}
        sum_count = sum(label_count.values())
        for k,v in label_count.items():
            self.label_weight[k] = v/sum_count

    def __len__(self):
        return len(self.data)

    # 找到分箱编号
    def find_bin(self, value, intervals):
        v = round(value, 2)
        for i, (s, e) in enumerate(intervals):
            if s <= v <= e:
                return i
        return len(intervals) - 1

    def field_to_token(self, field_name, bin_idx):
        """将字段的 bin idx 映射到全局 token id（避免字段冲突）"""
        return self.field_offsets[field_name] + bin_idx

    def __getitem__(self, idx):

        item = self.data[idx]
        label_str = self.labels[idx]
        label_int = self.class_mapping[label_str]

        # =====================================================
        # Step 1: label_token 前缀
        # =====================================================
        seq_tokens = []
        field_ids = []

        label_token = self.field_to_token("label_token", label_int)
        seq_tokens.append(label_token)
        field_ids.append(self.field_list.index("label_token"))
        
        len_int = min(len(item["series"]),self.max_seq_len - 1)
        len_token = self.field_to_token("len_token", len_int)
        seq_tokens.append(len_token)
        field_ids.append(self.field_list.index("len_token"))
        
        # seq_tokens.append(self.field_to_token("sep_token", SEPS["LABEL_SEP"]))
        seq_tokens.append(self.field_to_token("begin_token", 0))
        field_ids.append(self.field_list.index("begin_token"))
        
        weight = self.label_weight[label_str] * len(self.label_weight)

        # =====================================================
        # Step 2: port prefix
        # =====================================================
        
            
        # seq_tokens.append(self.field_to_token("sep_token", SEPS["PREFIX_SEP"]))
        # seq_tokens.append(self.field_to_token("pre_token", 0))
        # field_ids.append(self.field_list.index("pre_token"))  

        # =====================================================
        # Step 3: packet fields 展平
        # =====================================================
        pkt_count = 0
        for pkt in item["series"]:
            if pkt_count >= self.max_seq_len:
                break

            for attr in self.sery_attrs:
                # bin_id = self.find_bin(pkt[attr], self.bins_data[attr]["intervals"])
                if attr == "time":
                    token_ids = list(struct.pack(">d", pkt[attr]))
                    tokens = [self.field_to_token(attr,token_id) for token_id in token_ids]
                elif attr == "pkt_len":
                    plen = abs(int(pkt[attr]))
                    plen = max(-1500, min(plen, 1500))
                    # direction = 0 if pkt["pkt_len"] > 0 else 1
                    # token_ids = list(struct.pack(">H", plen)) + [direction]
                    token_id = (plen + 1500)
                    # tokens = [self.field_to_token(attr,token_id) for token_id in token_ids]
                    tokens = [self.field_to_token(attr,token_id)]
                else:
                    token_id = max(0, min(int(pkt.get(attr, 0)), 255))
                    tokens = [self.field_to_token(attr,token_id)]
                    
                # token = self.field_to_token(attr, bin_id)

                seq_tokens += tokens
                field_ids += [self.field_list.index(attr)] * len(tokens)
                
            # seq_tokens.append(self.field_to_token("sep_token", SEPS["PACKET_SEP"]))
            seq_tokens.append(self.field_to_token("sep_token", SEPS["SEP"]))
            field_ids.append(self.field_list.index("sep_token")) 

            pkt_count += 1
        
        # seq_tokens.append(self.field_to_token("sep_token", SEPS["SUFFIX_SEP"]))
        seq_tokens.append(self.field_to_token("suf_token", 0))
        field_ids.append(self.field_list.index("suf_token"))  

        # =====================================================
        # Step 4: IP suffix
        # =====================================================
        for attr in self.port_attrs:
            # bin_id = self.find_bin(item[attr], self.bins_data[attr]["intervals"])
            # token = self.field_to_token(attr, bin_id)
            
            port = max(0, min(int(item[attr]), 65535))
            token_ids = list(struct.pack(">H", port))
            
            tokens = [self.field_to_token(attr,token_id) for token_id in token_ids]
            seq_tokens += tokens
            field_ids += [self.field_list.index(attr)] * len(tokens)

            # seq_tokens.append(token)
            # field_ids.append(self.field_list.index(attr))
            
        for attr in self.ip_attrs:
            token_ids = list(struct.pack(">I", int(item[attr])))
            
            tokens = [self.field_to_token(attr,token_id) for token_id in token_ids]
            seq_tokens += tokens
            field_ids += [self.field_list.index(attr)] * len(tokens)
            # bin_id = self.find_bin(item[attr], self.bins_data[attr]["intervals"])
            # token = self.field_to_token(attr, bin_id)

            # seq_tokens.append(token)
            # field_ids.append(self.field_list.index(attr))
            
        # seq_tokens.append(self.field_to_token("sep_token", SEPS["EOS_SEP"]))
        seq_tokens.append(self.field_to_token("end_token", 0))
        field_ids.append(self.field_list.index("end_token"))  

        # =====================================================
        # Step 5: Padding 到固定长度
        # =====================================================
        max_len = cal_max_len(self.port_attrs,self.max_seq_len,self.sery_attrs,self.ip_attrs)

        pad_len = max_len - len(seq_tokens)
        if pad_len > 0:
            seq_tokens += [self.field_to_token("padding_token", 0)] * pad_len
            field_ids += [self.field_list.index("padding_token")] * pad_len  

        seq_tokens = torch.tensor(seq_tokens, dtype=torch.long)
        field_ids = torch.tensor(field_ids, dtype=torch.long)
        attn_mask = (seq_tokens != self.field_to_token("padding_token", 0)).long()
        weight = torch.tensor(weight, dtype=torch.float32)

        return seq_tokens, field_ids, attn_mask, label_int, len_int, weight

def collate_fn(batch):
    seqs = []
    field_types = []
    masks = []
    labels = []
    lens = []
    weights = []

    for seq, field_id, attn_mask, label, len_int, weight in batch:
        seqs.append(seq)
        field_types.append(field_id)
        masks.append(attn_mask)
        labels.append(label)
        lens.append(len_int)
        weights.append(weight)

    seqs = torch.stack(seqs, dim=0)              # (B,S)
    field_types = torch.stack(field_types, dim=0)
    masks = torch.stack(masks, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    lens = torch.tensor(lens, dtype=torch.long)
    weights = torch.tensor(weights)

    # 对于自回归生成，target = 输入序列
    target = seqs.clone()

    return {
        "seq": seqs,
        "field_type": field_types,
        "mask": masks,
        "label": labels,
        "target": target,
        "len": lens,
        "weight": weights
    }
