import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import os

class CustomDataset(Dataset):
    def __init__(self, data_folder, bins_file, class_mapping, max_seq_len, meta_attrs, sery_attrs, transform=None):

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
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        def find_interval(value, intervals):
            for idx, [start, end] in enumerate(intervals):
                if start <= round(value,2) <= end:
                    return idx 
            if round(value,2) > intervals[-1][1]:
                return len(intervals) - 1
            if round(value,2) < intervals[0][0]:
                return 0
            return None
            
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
            meta_id = find_interval(meta,self.bins_data[attr]['intervals'])
            meta_list.append(meta_id)
            
        count = 0
        for pkt in item['series']:
            id_list = []
            for attr in self.sery_attrs:
                id_list.append(find_interval(pkt[attr],self.bins_data[attr]['intervals']))
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
