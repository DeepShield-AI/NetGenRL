import json
import random
from post_process.check_bestmodel import get_real_data, pad
from transformer.transformer import TrafficTransformer
from transformer.dataset import FlattenTransformerDataset, collate_fn
# from transformer.generate import generate_tokens
from transformer.config import *
from torch.utils.data import DataLoader
import torch
# import numpy as np
# import ot
import struct

class Tokenizer:
    def __init__(self, field_list, field_vocab_sizes, seps_dict):
        """
        field_list: 顺序固定的字段列表（训练时已固定）
        field_vocab_sizes: dict[field] = vocab_size
        seps_dict: SEPS 字典（如 {"LABEL_SEP": 1, ...}）
        """

        self.field_list = field_list
        self.field_vocab_sizes = field_vocab_sizes  # dict: field -> N
        self.seps_dict = seps_dict

        # ==========================================================
        # 1. 计算每个 field 的 vocabulary 起始 offset
        # ==========================================================
        self.field_starts = {}
        offset = 0
        for field in field_list:
            self.field_starts[field] = offset
            offset += field_vocab_sizes[field]

        self.total_vocab_size = offset

        # build token → field id 反查表
        self.token_to_field_cache = []
        for field in field_list:
            start = self.field_starts[field]
            size = field_vocab_sizes[field]
            self.token_to_field_cache += [field] * size

        # field_name → index（用于模型输入 field_ids）
        self.field_name_to_index = {name: i for i, name in enumerate(self.field_list)}

    def field_to_token(self, field, bin_id):
        return self.field_starts[field] + bin_id
    
    def token_to_field(self, token_id):
        return self.token_to_field_cache[token_id]
    
    def token_to_field_id(self, token_id):
        field = self.token_to_field(token_id)
        return self.field_name_to_index[field]

    def token_to_field_and_value(self, token_id):
        field = self.token_to_field(token_id)
        bin_id = token_id - self.field_starts[field]
        return field, bin_id
    
    def token_to_value(self, token_id):
        field = self.token_to_field(token_id)
        bin_id = token_id - self.field_starts[field]
        return bin_id
    
    def get_field_vocab_range(self, field):
        start = self.field_starts[field]
        end = start + self.field_vocab_sizes[field]
        return start, end

    def is_sep(self, token, sep_name):
        sep_token = self.field_to_token("sep_token", self.seps_dict[sep_name])
        return token == sep_token

def save_data(label, data, meta_attrs, sery_attrs, result_folder_name):
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
        
    with open(result_folder_name + label + '.json','w') as file:
        json.dump(json_data,file)
        
    print(f"Save data to {result_folder_name + label + '.json'}")

# def sample_flow(flow, bins_data, meta_attrs, sery_attrs):
#     final_seq = []
    
#     metas = []
#     for i, meta_attr in enumerate(meta_attrs):
#         attr = round(random.uniform(bins_data[meta_attr]['intervals'][flow[i]][0], bins_data[meta_attr]['intervals'][flow[i]][1]))
#         metas.append(attr)
    
#     for i in range(len(meta_attrs),len(flow)):
#         pkt = []
#         for j, sery_attr in enumerate(sery_attrs):
#             if j == 0:
#                 attr = round(random.uniform(bins_data[sery_attr]['intervals'][flow[i]][0], bins_data[sery_attr]['intervals'][flow[i]][1]),2)
#             else:
#                 attr = round(random.uniform(bins_data[sery_attr]['intervals'][flow[i]][0], bins_data[sery_attr]['intervals'][flow[i]][1]))
#             pkt.append(attr)
#         final_seq.append(pkt + metas)
        
#     return final_seq     

def tokens_to_bins(seq_tokens, tokenizer, port_attrs, sery_attrs, ip_attrs, label_dict):
    ip_bins = []
    port_bins = []
    pkt_bins = []
    label_list = list(label_dict.keys())

    cur_pkt = []
    
    begin = False
    suffix = True
    
    id = 0
    while id < len(seq_tokens):
        token = seq_tokens[id]
    # for token in seq_tokens:
        if tokenizer.token_to_field(token) == "padding_token":
            # padding，忽略
            continue

        field, bin_id = tokenizer.token_to_field_and_value(token)

        # 终止符和分隔符
        # if tokenizer.is_sep(token, "EOS_SEP"):
        #     # EOS 结束
        #     if cur_pkt:
        #         pkt_bins.append(cur_pkt)
        #     break
        # elif tokenizer.is_sep(token, "PACKET_SEP"):
        #     if cur_pkt:
        #         pkt_bins.append(cur_pkt)
        #         cur_pkt = []
        #     continue
        # elif tokenizer.is_sep(token, "LABEL_SEP") or tokenizer.is_sep(token, "PREFIX_SEP") or tokenizer.is_sep(token, "SUFFIX_SEP"):
        #     continue
        
        if field == "label_token":
            label = label_list[bin_id]
        
        if tokenizer.token_to_field(token) == "begin_token":
            begin = True
            id += 1
            continue
        
        if not begin:
            id += 1
            continue
        
        if tokenizer.token_to_field(token) == "suf_token":
            suffix = True
            id += 1
            # src_port = struct.unpack(">H", bytes(seq_tokens[id:id+2]))[0]
            src_port = struct.unpack(">H", bytes([tokenizer.token_to_value(seq_token) for seq_token in seq_tokens[id:id+2]]))[0]
            id += 2
            # dst_port = struct.unpack(">H", bytes(seq_tokens[id:id+2]))[0]
            dst_port = struct.unpack(">H", bytes([tokenizer.token_to_value(seq_token) for seq_token in seq_tokens[id:id+2]]))[0]
            id += 2
            # src_ip = struct.unpack(">I", bytes(seq_tokens[id:id+4]))[0]
            src_ip = struct.unpack(">I", bytes([tokenizer.token_to_value(seq_token) for seq_token in seq_tokens[id:id+4]]))[0]
            id += 4
            # dst_ip = struct.unpack(">I", bytes(seq_tokens[id:id+4]))[0]
            dst_ip = struct.unpack(">I", bytes([tokenizer.token_to_value(seq_token) for seq_token in seq_tokens[id:id+4]]))[0]
            id += 4
            port_bins = [src_port,dst_port]
            ip_bins = [src_ip,dst_ip]
            continue
            
        if suffix and tokenizer.token_to_field(token) == "end_token":
            break
            
        if tokenizer.token_to_field(token) == "sep_token":
            pkt_bins.append(cur_pkt)
            cur_pkt = []
            id += 1
            continue
            
        # time = struct.unpack(">d", bytes(seq_tokens[id:id+8]))[0]
        time = struct.unpack(">d", bytes([tokenizer.token_to_value(seq_token) for seq_token in seq_tokens[id:id+8]]))[0]
        id += 8
        # pkt_len = struct.unpack(">H", bytes(seq_tokens[id:id+2]))[0]
        # pkt_len = struct.unpack(">H", bytes([tokenizer.token_to_value(seq_token) for seq_token in seq_tokens[id:id+2]]))[0]
        # id += 1
        # direction = tokenizer.token_to_value(seq_tokens[id])
        pkt_len = tokenizer.token_to_value(seq_tokens[id]) - 1500
        id += 1
        # if direction > 0:
        #     pkt_len = -pkt_len
            
        flags = tokenizer.token_to_value(seq_tokens[id])
        id += 1
        
        ttl = tokenizer.token_to_value(seq_tokens[id])
        id += 1
        
        cur_pkt = [time,pkt_len,flags,ttl]
        
        # if tokenizer.token_to_field(token) == "sep_token":
        #     if len(cur_pkt) > 0:
        #         pkt_bins.append(cur_pkt)
        #         cur_pkt = []
        #     continue
        
            

        # 根据字段归类
        
        # elif field in port_attrs:
        #     port_bins.append(bin_id)
        # elif field in sery_attrs:
        #     cur_pkt.append(bin_id)
        # elif field in ip_attrs:
        #     ip_bins.append(bin_id)

    return label, port_bins + ip_bins + pkt_bins

def get_gen_data(dataset, model, bins_data, sery_attrs, port_attrs, ip_attrs, batch_size, device, tokenizer, label_dict):
    meta_attrs = port_attrs + ip_attrs
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    generated_sequences = {}
    with torch.no_grad():
        for id, batch in enumerate(dataloader):
            ftype = batch["field_type"].to(device)
            label = batch["label"].to(device)
            lengths = batch["len"].to(device)
            seq = batch["seq"]
            
        
            fake_data = model.generate(ftype,label,lengths,1.0)
            # 将生成结果按序列长度截断
            for i, length in enumerate(lengths):
                # print(seq[i])
                token_len = cal_max_len(port_attrs,length + 1,sery_attrs,ip_attrs)
                flow_tokens = fake_data[i, :token_len].cpu().tolist()
                print(flow_tokens)
                label, flow = tokens_to_bins(flow_tokens,tokenizer, port_attrs, sery_attrs, ip_attrs,label_dict)
                print(flow)
                # generated_sequences.append(flow)
                
                pkt_seq = []
                metas = flow[:len(meta_attrs)]
                for i in range(len(meta_attrs),len(flow)):
                    pkt = flow[i]
                    pkt = pkt + metas
                    pkt_seq.append(pkt)
                
                if label not in generated_sequences:
                    generated_sequences[label] = []
                # generated_sequences[label].append(flow)
                generated_sequences[label].append(pkt_seq)
            print(id,"/",int(len(dataset)/batch_size) + 1)
    # print(generated_sequences)
    return generated_sequences
    # final_seq = {}
    
    # for label, flows in generated_sequences.items():
    #     final_seq[label] = []
    #     for flow in flows:
    #         pkt_seq = []
    #         # metas = []
    #         # for i, meta_attr in enumerate(meta_attrs):
    #         #     attr = round(random.uniform(bins_data[meta_attr]['intervals'][flow[i]][0], bins_data[meta_attr]['intervals'][flow[i]][1]))
    #         #     metas.append(attr)
    #         metas = flow[:len(meta_attrs)]
    #         for i in range(len(meta_attrs),len(flow)):
    #             pkt = flow[i]
    #             pkt = pkt + metas
    #             pkt_seq.append(pkt)
    
            
    #         # for i in range(len(meta_attrs),len(flow)):
    #         #     pkt = []
    #         #     for j, sery_attr in enumerate(sery_attrs):
    #         #         if j == 0:
    #         #             attr = round(random.uniform(bins_data[sery_attr]['intervals'][flow[i][j]][0], bins_data[sery_attr]['intervals'][flow[i][j]][1]),2)
    #         #         else:
    #         #             attr = round(random.uniform(bins_data[sery_attr]['intervals'][flow[i][j]][0], bins_data[sery_attr]['intervals'][flow[i][j]][1]))
    #         #         pkt.append(attr)
    #         #     pkt = pkt + metas
    #         #     pkt_seq.append(pkt)
    #         final_seq[label].append(pkt_seq)
    # return final_seq  

def generate_data(label_dict, dataset, json_folder, bins_folder, model_folder, result_folder, port_attrs, ip_attrs, sery_attrs, batch_size, max_seq_len, expand_times):
    data_folder = f'./{json_folder}/{dataset}/'
    bins_file_name = f'./{bins_folder}/bins_{dataset}.json'
    result_folder_name = f'./{result_folder}/{dataset}/'
    meta_attrs = port_attrs + ip_attrs

    bins_data = {}
    with open(bins_file_name, 'r') as f_bin:
        bins_data = json.load(f_bin)

    real_datas = get_real_data(data_folder,label_dict,meta_attrs,sery_attrs,bins_data,max_seq_len)
    
    # model_name = save_folder + f'generator_{model_id}.pth'
    model_folder_name = f'./{model_folder}/{dataset}/'
    model_name = f'{model_folder_name}/transformer_best.pth'
    
    
        
    # fake_datas = {}
    # # times = 20
    # for label, data in real_datas.items():
    #     fake_flows = generate_tokens(label_dict, dataset, json_folder, bins_folder, model_folder, port_attrs,ip_attrs,sery_attrs,model_paras, label, len(data) * (expand_times + 1))
    #     fake_data = []
    #     for token_seq in fake_flows:
    #         flow = sample_flow(token_seq, bins_data)
    #         fake_data.append(flow)
    #     fake_datas[label] = fake_data
    #     print("Generate data of", label)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = FlattenTransformerDataset(
        data_folder= data_folder,
        bins_file=bins_file_name,
        class_mapping=label_dict,
        max_seq_len=max_seq_len,
        port_attrs=port_attrs,
        ip_attrs=ip_attrs,
        sery_attrs=sery_attrs
    )

    max_len = cal_max_len(port_attrs,max_seq_len,sery_attrs,ip_attrs)
    model = TrafficTransformer(
        field_vocab_sizes=ds.field_vocab_sizes,
        field_emb_dim=EMBED_DIM,
        d_model=EMBED_DIM,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ffn_dim=FFN_DIM,
        dropout=DROPOUT,
        max_len=max_len,
        label_size=len(label_dict),
        label_emb_dim=LABEL_EMB_DIM,
        max_seq_len=max_seq_len,
        len_emb_dim=LEN_EMB_DIM
    ).to(device)
    
    ckpt = torch.load(model_name, map_location=device)
    model.load_state_dict(ckpt["state"])
    
    tokenizer = Tokenizer(
        field_list=model.field_names,
        field_vocab_sizes=model.field_vocab_sizes,
        seps_dict=SEPS
    )
    
    fake_datas = get_gen_data(ds,model,bins_data,sery_attrs,port_attrs,ip_attrs,batch_size,device,tokenizer,label_dict)
    
    # ot_sum = 0
    # for label in label_dict.keys():
    #     real_data = real_datas[label]
    #     fake_data = fake_datas[label]
    #     real_sequences = np.array([pad(seq, max_seq_len) for seq in real_data])         # Shape: (num_samples, seq_len, num_dims)
    #     generated_sequences = np.array([pad(seq, max_seq_len) for seq in fake_data]) 

    #     X = real_sequences
    #     Y = generated_sequences
        
    #     X_filled = np.nan_to_num(X, nan=1)  # 或者 nan=0，如果认为0不影响
    #     Y_filled = np.nan_to_num(Y, nan=1)
    
    #     cost_matrix = np.linalg.norm(X_filled[:, None, :, :] - Y_filled[None, :, :, :], axis=(-2, -1))
        
    #     ot_distance = ot.emd2([], [], cost_matrix) 
            
    #         # print(f"label {label}: {ot_distance}")
            
    #     ot_sum += ot_distance
        
    # print(f"model OT:", ot_sum)
    
    # print(fake_datas)
    for label, data in fake_datas.items():
        save_data(label,data,meta_attrs,sery_attrs,result_folder_name)
        
