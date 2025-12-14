import numpy as np
import torch
# from seqCGAN.generator import Generator  # 假设你有一个定义好的 Discriminator 类
from stateFormer.transformer import TransGenerator  # 假设你有一个定义好的 Discriminator 类
import json
import random
import math
from torch.utils.data import DataLoader, Dataset
import os
import ot

  
def get_real_data(data_folder, label_dict, meta_attrs, sery_attrs, bins_data, max_seq_len):
    data_dic = {}
    for filename in os.listdir(data_folder):
        with open(data_folder + filename, 'r') as f:
            if filename.split('.')[0] not in label_dict.keys():
                continue
            json_data = json.load(f)
            data_dic[filename.split('.')[0]] = []
            for item in json_data:
                meta_list = []
                for meta_attr in meta_attrs:
                    meta_list.append(item[meta_attr]/bins_data[meta_attr]['intervals'][-1][1])
                count = 0
                seq = []
                for pkt in item['series']:
                    attr_list = []
                    for sery_attr in sery_attrs:
                        attr_list.append(pkt[sery_attr]/bins_data[sery_attr]['intervals'][-1][1])
                    seq.append(attr_list + meta_list)
                    # seq.append(attr_list)
                    count += 1
                    if count >= max_seq_len:
                        break
                data_dic[filename.split('.')[0]].append(seq)
    return data_dic

class SequenceDataset(Dataset):
    def __init__(self, datas, label_str, label_dict, label_dim):
        """
        :param sequences: 一个包含真实序列的列表，每个序列是一个 ndarray 或 list
        """
        self.datas = datas
        self.lengths = [len(seq) for seq in datas]  # 提取序列长度
        label_id = label_dict[label_str]
        self.label_one_hot = torch.zeros(label_dim).to('cpu') 
        self.label_one_hot[label_id] = 1

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.lengths[idx],self.label_one_hot
    
def get_fake_data(label_dict, label_dim, real_data, label_str, generator, bins_data, sery_attrs, meta_attrs, batch_size):
    seq_attrs = sery_attrs + meta_attrs
    dataset = SequenceDataset(real_data,label_str,label_dict,label_dim)
    dataloader = DataLoader(dataset, batch_size, shuffle=False)
    generated_sequences = []
    with torch.no_grad():
        for lengths, labels in dataloader:
            lengths = lengths.to(torch.device("cpu"))  # 确保在同一个设备上
            labels = labels.to(torch.device("cpu"))
            # print(lengths.shape)
            # print(labels.shape)
            batch_size = lengths.size(0)
            # 生成随机噪声向量
            # noise = torch.randn(len(lengths), noise_dim)
            # 输入生成器生成数据
            fake_data = generator.sample(batch_size,labels,lengths)
            # 将生成结果按序列长度截断
            for i, length in enumerate(lengths):
                generated_sequences.append(fake_data[i, :length].cpu().tolist())
       
    final_seqs = []         
    for seq in generated_sequences:
        f_seq = []
        for i in range(len(seq)):
            pkt = []
            for j,attr_id in enumerate(seq[i]): 
                if j == 0:
                    attr = round(random.uniform(bins_data[seq_attrs[j]]['intervals'][attr_id][0], bins_data[seq_attrs[j]]['intervals'][attr_id][1]),2) / bins_data[seq_attrs[j]]['intervals'][-1][1]
                elif j < len(sery_attrs) or i == 0:
                    attr = round(random.uniform(bins_data[seq_attrs[j]]['intervals'][attr_id][0], bins_data[seq_attrs[j]]['intervals'][attr_id][1])) / bins_data[seq_attrs[j]]['intervals'][-1][1]
                else:
                    attr = f_seq[0][j]
                
                pkt.append(attr)
            f_seq.append(pkt)
        final_seqs.append(f_seq)
    return final_seqs 

def pad(sequence, target_length, pad_value=np.nan):
    seq_len = len(sequence)
    if seq_len < target_length:
        padding = [[pad_value] * len(sequence[0])] * (target_length - seq_len)
        return sequence + padding  # 填充
    return sequence

def check_models(label_dict, dataset, json_folder, bins_folder, model_folder, meta_attrs, sery_attrs, batch_size, max_seq_len, checkpoint, epochs): 
    label_dim = len(label_dict)
    save_folder = f'./{model_folder}/{dataset}/'
    data_folder = f'./{json_folder}/{dataset}/'
    bins_file_name = f'./{bins_folder}/bins_{dataset}.json'
    seq_dim = len(meta_attrs) + len(sery_attrs)

    bins_data = {}
    with open(bins_file_name, 'r') as f_bin:
        bins_data = json.load(f_bin)
        x_list = [len(bins_d["intervals"]) for bins_d in bins_data.values()]

    real_datas = get_real_data(data_folder,label_dict,meta_attrs,sery_attrs,bins_data,max_seq_len)
    
    best_model_id = 0
    min_ot = math.inf

    for model_id in range(checkpoint,epochs + checkpoint,checkpoint):
        model_name = save_folder + f'generator_{model_id}.pth'
        # model_name = save_folder + f'generator_pre.pth'
        
        generator = TransGenerator(label_dim,seq_dim,max_seq_len,x_list,'cpu')
        checkpoint = torch.load(model_name, map_location=torch.device('cpu'))  # 加载保存的权重字典
        generator.load_state_dict(checkpoint)  # 将权重字典加载到模型中
        generator.eval()
        
        fake_datas = {}
        for label, data in real_datas.items():
            fake_data = get_fake_data(label_dict,label_dim,data,label,generator,bins_data,sery_attrs,meta_attrs,batch_size)
            fake_datas[label] = fake_data
            
        ot_sum = 0
        hm_sum = 0
        for label in label_dict.keys():
            real_data = real_datas[label]
            fake_data = fake_datas[label]
            real_sequences = np.array([pad(seq, max_seq_len) for seq in real_data])         # Shape: (num_samples, seq_len, num_dims)
            generated_sequences = np.array([pad(seq, max_seq_len) for seq in fake_data]) 

            X = real_sequences
            Y = generated_sequences
        
            X_filled = np.nan_to_num(X, nan=1)  # 或者 nan=0，如果认为0不影响
            Y_filled = np.nan_to_num(Y, nan=1)
    
            cost_matrix = np.linalg.norm(X_filled[:, None, :, :] - Y_filled[None, :, :, :], axis=(-2, -1))
            cost_matrix_ham = np.zeros((X_filled.shape[0], Y_filled.shape[0]))
            for i in range(X_filled.shape[0]):
                for j in range(Y_filled.shape[0]):
                    hamming_distance = np.sum(X_filled[i] != Y_filled[j])
                    cost_matrix_ham[i, j] = hamming_distance
        
            ot_distance = ot.emd2([], [], cost_matrix) 
            hm_distance = ot.emd2([], [], cost_matrix_ham) 
            
            # print(f"label {label}: {ot_distance}")
            
            ot_sum += ot_distance
            hm_sum += hm_distance
        
        print(f"model {model_id} OT:", ot_sum, "HM:",hm_sum)
            
        if min_ot > ot_sum:
            min_ot = ot_sum
            best_model_id = model_id
        
    print(f"Best model is {best_model_id}.")
    return best_model_id
    
    


    


