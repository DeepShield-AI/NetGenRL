# %%
from gensim.models import Word2Vec
import numpy as np
import json
import struct
import os

# %%
dataset = 'iscx'
data_fold = '../data/' + dataset + '/'
bins_file = f'../bins/bins_{dataset}.json'
word_vec_json_file = f'../wordvec/word_vec_{dataset}.json'
IP_ATTRIBUTE_LIST = ['src_ip','dst_ip']
PORT_ATTRIBUTE_LIST = ['src_port','dst_port']
SERY_ATTRIBUTE_LIST = ['time','pkt_len','flags','ttl']
max_seq_len = 16
# NPRINT_LINE_LEN = 114

# %%
with open(bins_file, 'r') as f_bin:
    bins_data = json.load(f_bin)

# pkt_len_intervals = [bins['intervals'] for bins in bins_data['packet_len']]
# time_intervals = [bins['intervals'] for bins in bins_data['time']]
# pkt_len_intervals = bins_data['packet_len']['intervals']
# time_intervals = bins_data['time']['intervals']

# %%
def get_seqs_meta(data_fold):
    json_data_dic = {}
    for filename in os.listdir(data_fold):
        with open(data_fold + filename, 'r') as f:
            json_data = json.load(f)
            json_data_dic[filename.split('.')[0]] = json_data
        
    def find_interval(value, intervals):
        for idx, [start, end] in enumerate(intervals):
            if start <= round(value,2) <= end:
                return idx  # 返回所在区间的下标
        print(value, intervals)
        return None
        
    seq_dic = {}
    
    for attribute in SERY_ATTRIBUTE_LIST:
        seq_dic[attribute] = []
    
    meta_list = []
    for data in json_data_dic.values():
        for idx, item in enumerate(data):
            for attribute in SERY_ATTRIBUTE_LIST:
                attr_seq = []
                for i, pkt in enumerate(item['series']):
                    if i >= max_seq_len:
                        break
                    attr_seq.append(find_interval(pkt[attribute], bins_data[attribute]['intervals']))
                seq_dic[attribute].append(attr_seq)
            
        for item in data:
            meta = []
            for attribute in PORT_ATTRIBUTE_LIST + IP_ATTRIBUTE_LIST:
                # meta.append(item[attribute])
                meta.append(find_interval(item[attribute], bins_data[attribute]['intervals']))
            meta_list.append(meta)
        
    return seq_dic, meta_list

# %%
def get_wv_model(seqs, vector_size, window, min_count = 1, sg = 1):
    seq_str = [[str(c) for c in seq] for seq in seqs]
    model = Word2Vec(sentences=seq_str, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    return model

def get_meta_model(seqs, vector_size, window, min_count = 1, sg = 1):
    seq_str = [[str(c)+f'_{i}' for i,c in enumerate(seq)] for seq in seqs]
    model = Word2Vec(sentences=seq_str, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    return model


# %%
def run_wv():
    sequences_dic, meta_list = get_seqs_meta(data_fold)

    model_dic = {}

    for key, seq in sequences_dic.items():
        model_dic[key] = get_wv_model(seq,8,5)
        model_dic[key].save(f"../wordvec/{key}.model")
    
    meta_model = get_meta_model(meta_list,4,3)
    meta_model.save(f"../wordvec/meta.model")

    set = {}
    
    for key, seqs in sequences_dic.items():
        set[key] = []
        for seq in seqs:
            for v in seq:
                if v not in set[key]:
                    if v == None:
                        print(seq)
                    set[key].append(v)
        set[key] = sorted(set[key])
        print(key,len(set[key]))
    
    meta_attrs = PORT_ATTRIBUTE_LIST + IP_ATTRIBUTE_LIST
    for key in meta_attrs:
        set[key] = []
    for meta in meta_list:
        for i, v in enumerate(meta):
            if v not in set[meta_attrs[i]]:
                set[meta_attrs[i]].append(v)
    for key in meta_attrs:
        set[key] = sorted(set[key])
        print(key,len(set[key]))

    word_vec_metrics = {}
    count = 0
    for key, seq in set.items():
        word_vec_metrics[key] = []
        for i in seq:
            if key in model_dic:
                word_vec_metrics[key].append(model_dic[key].wv[str(i)].tolist())
            else:
                word_vec_metrics[key].append(meta_model.wv[str(i)+f'_{count}'].tolist())
        if key not in model_dic:
            count += 1
            

    json_str = json.dumps(word_vec_metrics)
    with open(word_vec_json_file, 'w') as file:
        file.write(json_str)

if __name__ == '__main__':
    run_wv()
