import ot
import numpy as np
import torch
from seqCGAN.generator import Generator  # 假设你有一个定义好的 Discriminator 类
import json
import random
from post_process.check_bestmodel import SequenceDataset
from torch.utils.data import DataLoader

MAX_SEQ_LEN = 16

def pad(sequence, target_length, pad_value=np.nan):
    seq_len = len(sequence)
    if seq_len < target_length:
        padding = [[pad_value] * len(sequence[0])] * (target_length - seq_len)
        return sequence + padding 
    return sequence


def cal_euler(real_datas, fake_datas, label_dict):
    ot_sum = 0
    for label in label_dict.keys():
        real_data = real_datas[label]
        fake_data = fake_datas[label]
        real_sequences = np.array([pad(seq, MAX_SEQ_LEN) for seq in real_data])         # Shape: (num_samples, seq_len, num_dims)
        generated_sequences = np.array([pad(seq, MAX_SEQ_LEN) for seq in fake_data]) 

        X = real_sequences
        Y = generated_sequences
        
        X_filled = np.nan_to_num(X, nan=1)
        Y_filled = np.nan_to_num(Y, nan=1)
    
        cost_matrix = np.linalg.norm(X_filled[:, None, :, :] - Y_filled[None, :, :, :], axis=(-2, -1))
        
        ot_distance = ot.emd2([], [], cost_matrix) 
            
        ot_sum += ot_distance
    
    return ot_sum
        # print(ot_distance)
        
def cal_hamming(real_datas, fake_datas, label_dict):
    hamming_sum = 0
    for label in label_dict.keys():
        real_data = real_datas[label]
        fake_data = fake_datas[label]
        real_sequences = np.array([pad(seq, MAX_SEQ_LEN) for seq in real_data])         # Shape: (num_samples, seq_len, num_dims)
        generated_sequences = np.array([pad(seq, MAX_SEQ_LEN) for seq in fake_data]) 

        X = real_sequences
        Y = generated_sequences
        
        X_filled = np.nan_to_num(X, nan=-1) 
        Y_filled = np.nan_to_num(Y, nan=-1)
    
        cost_matrix = np.zeros((X_filled.shape[0], Y_filled.shape[0]))

        for i in range(X_filled.shape[0]):
            for j in range(Y_filled.shape[0]):
                hamming_distance = np.sum(X_filled[i] != Y_filled[j])
                cost_matrix[i, j] = hamming_distance

        hamming_distance = ot.emd2([], [], cost_matrix) 
        hamming_sum += np.sum(hamming_distance)
        
    return hamming_sum

def data2json(data, meta_attrs, sery_attrs):
    json_data = []
    for seq in data:
        item = {}
        for i, meta_attr in enumerate(meta_attrs):
            item[meta_attr] = seq[0][i+len(sery_attrs)]
        sery = []
        for pkt in seq:
            p = {}
            for i, sery_attr in enumerate(sery_attrs):
                p[sery_attr] = pkt[i]
            sery.append(p)
        item['series'] = sery
        json_data.append(item)
        
    return json_data

def get_fake_data(label_dict, label_dim, real_data, label_str, generator, bins_data, sery_attrs, meta_attrs, batch_size):
    seq_attrs = sery_attrs + meta_attrs
    dataset = SequenceDataset(real_data,label_str,label_dict,label_dim)
    dataloader = DataLoader(dataset, batch_size, shuffle=False)
    generated_sequences = []
    with torch.no_grad():
        for lengths, labels in dataloader:
            lengths = lengths.to(generator.device)  # 确保在同一个设备上
            labels = labels.to(generator.device)
            batch_size = lengths.size(0)
            fake_data = generator.sample(batch_size,labels,lengths)
            for i, length in enumerate(lengths):
                generated_sequences.append(fake_data[i, :length].cpu().tolist())
       
    final_seqs = []         
    for seq in generated_sequences:
        f_seq = []
        for i in range(len(seq)):
            pkt = []
            for j,attr_id in enumerate(seq[i]): 
                if j == 0:
                    attr = round(random.uniform(bins_data[seq_attrs[j]]['intervals'][attr_id][0], bins_data[seq_attrs[j]]['intervals'][attr_id][1]),2)/bins_data[seq_attrs[j]]['intervals'][-1][1]
                elif j < len(sery_attrs) or i == 0:
                    attr = round(random.uniform(bins_data[seq_attrs[j]]['intervals'][attr_id][0], bins_data[seq_attrs[j]]['intervals'][attr_id][1]))/bins_data[seq_attrs[j]]['intervals'][-1][1]
                else:
                    attr = f_seq[0][j]
                
                pkt.append(attr)
            f_seq.append(pkt)
        final_seqs.append(f_seq)
    return final_seqs    

def get_fake_json(label_dict, label_dim, real_datas, generator, bins_data, sery_attrs, meta_attrs, batch_size):
    fake_datas = {}
    for label, data in real_datas.items():
        fake_data = get_fake_data(label_dict,label_dim,data,label,generator,bins_data,sery_attrs,meta_attrs,batch_size)
        fake_datas[label] = fake_data
    return fake_datas