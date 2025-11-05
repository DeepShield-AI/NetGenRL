# %%
from gensim.models import Word2Vec
import json
import os

# %%
def get_seqs_meta(data_fold, ip_attrs, port_attrs, sery_attrs, max_seq_len, bins_data):
    json_data_dic = {}
    for filename in os.listdir(data_fold):
        with open(data_fold + filename, 'r') as f:
            json_data = json.load(f)
            json_data_dic[filename.split('.')[0]] = json_data
        
    def find_interval(value, intervals):
        for idx, [start, end] in enumerate(intervals):
            if start <= round(value,2) <= end:
                return idx  # 返回所在区间的下标
        if round(value,2) > intervals[-1][1]:
            return len(intervals) - 1
        print(value, intervals)
        return None
        
    seq_dic = {}
    
    for attribute in sery_attrs:
        seq_dic[attribute] = []
    
    meta_list = []
    for data in json_data_dic.values():
        for idx, item in enumerate(data):
            for attribute in sery_attrs:
                attr_seq = []
                for i, pkt in enumerate(item['series']):
                    if i >= max_seq_len:
                        break
                    attr_seq.append(find_interval(pkt[attribute], bins_data[attribute]['intervals']))
                seq_dic[attribute].append(attr_seq)
            
        for item in data:
            meta = []
            for attribute in port_attrs + ip_attrs:
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

def run_word_vec(dataset,json_folder,bins_folder,wordvec_folder, ip_attrs, port_attrs, sery_attrs, max_seq_len, series_word_vec_size, meta_word_vec_size):
    data_fold = f'./{json_folder}/{dataset}/'
    bins_file = f'./{bins_folder}/bins_{dataset}.json'
    word_vec_json_file = f'./{wordvec_folder}/word_vec_{dataset}.json'

# %%
    with open(bins_file, 'r') as f_bin:
        bins_data = json.load(f_bin)
        
    
    sequences_dic, meta_list = get_seqs_meta(data_fold,ip_attrs,port_attrs,sery_attrs,max_seq_len,bins_data)

    model_dic = {}

    for key, seq in sequences_dic.items():
        model_dic[key] = get_wv_model(seq,series_word_vec_size,5)
    
    meta_model = get_meta_model(meta_list,meta_word_vec_size,3)

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
    
    meta_attrs = port_attrs + ip_attrs
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
                # pass
        if key not in model_dic:
            count += 1
            

    json_str = json.dumps(word_vec_metrics)
    with open(word_vec_json_file, 'w') as file:
        file.write(json_str)
        