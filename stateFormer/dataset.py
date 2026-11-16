import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import os

class CustomDataset(Dataset):
    def __init__(self, data_folder, bins_file, class_mapping, max_seq_len, meta_attrs, sery_attrs, transform=None, preencode=False):

        self.data_folder = data_folder
        self.class_mapping = class_mapping
        self.max_seq_len = max_seq_len
        self.meta_attrs = meta_attrs
        self.sery_attrs = sery_attrs
        self.transform = transform
        
        self.data = []
        self.labels = []
        label_count = {}
        for filename in os.listdir(self.data_folder):
            with open(self.data_folder + filename, 'r') as f:
                if filename.split('.')[0] not in class_mapping.keys():
                    continue
                json_data = json.load(f)
                self.data += json_data
                label_count[filename.split('.')[0]] = 1/len(json_data)
                self.labels += [filename.split('.')[0]] * len(json_data)
        
        self.label_weight = {}
        sum_count = sum(label_count.values())
        for k,v in label_count.items():
            self.label_weight[k] = v/sum_count
            
        with open(bins_file, 'r') as f_bin:
            self.bins_data = json.load(f_bin)

        self._size = len(self.data)
        self._encoded = None
        self._lookup_cache = None
        if preencode:
            # Deterministic preprocessing only: preserve sample order and RNG.
            fields = len(self.sery_attrs) + len(self.meta_attrs)
            self._encoded = (
                torch.empty((self._size, self.max_seq_len, fields), dtype=torch.long),
                torch.empty((self._size, len(self.class_mapping)), dtype=torch.float32),
                torch.empty(self._size, dtype=torch.float32),
                torch.empty(self._size, dtype=torch.float32),
            )
            self._lookup_cache = {attr: {} for attr in self.sery_attrs + self.meta_attrs}
            for idx in range(self._size):
                for column, value in zip(self._encoded, self._encode_item(idx)):
                    column[idx] = value
            # The tensors retain only the training prefix; release raw JSON
            # objects and temporary lookup dictionaries before training.
            self._lookup_cache = None
            self.data = []
            self.labels = []

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        if self._encoded is not None:
            return tuple(column[idx] for column in self._encoded)
        return self._encode_item(idx)

    def _find_interval(self, attr, value):
        rounded = round(value, 2)
        cache = self._lookup_cache[attr] if self._lookup_cache is not None else None
        if cache is not None and rounded in cache:
            return cache[rounded]
        intervals = self.bins_data[attr]['intervals']
        result = None
        # Retain first-match semantics for shared boundaries and overlaps.
        for idx, (start, end) in enumerate(intervals):
            if start <= rounded <= end:
                result = idx + 1
                break
        if result is None:
            if rounded > intervals[-1][1]:
                result = len(intervals)
            elif rounded < intervals[0][0]:
                result = 1
            else:
                print(f"Value {value} not found in any interval.")
        if cache is not None and result is not None:
            cache[rounded] = result
        return result

    def _encode_item(self, idx):
        item = self.data[idx]
        label_str = self.labels[idx]
        length = min(len(item['series']),self.max_seq_len)
        label_int = self.class_mapping[label_str]
        
        weight = self.label_weight[label_str] * len(self.label_weight)
        labels_one_hot = F.one_hot(torch.tensor(label_int), num_classes=len(self.class_mapping)).float()

        seq = []
        
        meta_list = []
        for attr in self.meta_attrs:
            meta = item[attr]
            meta_id = self._find_interval(attr, meta)
            meta_list.append(meta_id)
            
        count = 0
        for pkt in item['series']:
            id_list = []
            for attr in self.sery_attrs:
                id_list.append(self._find_interval(attr, pkt[attr]))
                # print(attr)
            id_list += meta_list
            seq.append(id_list)
            count += 1
            if count >= self.max_seq_len:
                break
        
        if len(seq) < self.max_seq_len:
            seq = np.pad(seq, ((0, self.max_seq_len - len(seq)), (0, 0)), mode='constant', constant_values=0)
        
        # print(seq)
        seq = torch.tensor(seq, dtype=torch.long)
        length = torch.tensor(length, dtype=torch.float32)
        weight = torch.tensor(weight, dtype=torch.float32)
        
        return seq, labels_one_hot, length, weight
