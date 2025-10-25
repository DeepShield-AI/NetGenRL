import torch
from seqCGAN.generator import Generator  # 假设你有一个定义好的 Discriminator 类
import json
import random
from post_process.check_bestmodel import get_real_data, SequenceDataset
from torch.utils.data import DataLoader

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

def get_gen_data(label_dict, label_dim, real_data, label_str, generator, bins_data, sery_attrs, meta_attrs, batch_size):
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
                    attr = round(random.uniform(bins_data[seq_attrs[j]]['intervals'][attr_id][0], bins_data[seq_attrs[j]]['intervals'][attr_id][1]),2)
                elif j < len(sery_attrs) or i == 0:
                    attr = round(random.uniform(bins_data[seq_attrs[j]]['intervals'][attr_id][0], bins_data[seq_attrs[j]]['intervals'][attr_id][1]))
                else:
                    attr = f_seq[0][j]
                
                pkt.append(attr)
            f_seq.append(pkt)
        final_seqs.append(f_seq)
    return final_seqs         

def generate_data(label_dict, dataset, json_folder, bins_folder, wordvec_folder, model_folder, result_folder, meta_attrs, sery_attrs, batch_size, max_seq_len, checkpoint, model_id, expand_times):
    label_dim = len(label_dict)
    save_folder = f'./{model_folder}/{dataset}/'
    data_folder = f'./{json_folder}/{dataset}/'
    bins_file_name = f'./{bins_folder}/bins_{dataset}.json'
    wordvec_file_name = f'./{wordvec_folder}/word_vec_{dataset}.json'
    result_folder_name = f'./{result_folder}/{dataset}/'
    seq_dim = len(meta_attrs) + len(sery_attrs)
    with open(wordvec_file_name, 'r') as f:
        wv_dict = json.load(f)
    
    wv = {}
    for key, metrics in wv_dict.items():
        wv[key] = torch.tensor(metrics, dtype=torch.float32)
    
    x_list = [wv_tensor.size(0) for wv_tensor in wv.values()]

    bins_data = {}
    with open(bins_file_name, 'r') as f_bin:
        bins_data = json.load(f_bin)

    real_datas = get_real_data(data_folder,label_dict,meta_attrs,sery_attrs,bins_data,max_seq_len)
    
    # model_name = save_folder + f'generator_{model_id}.pth'
    model_name = save_folder + f'generator_pre.pth'
        
    generator = Generator(label_dim,seq_dim,max_seq_len,x_list,'cpu')
    checkpoint = torch.load(model_name, map_location=torch.device('cpu'))  # 加载保存的权重字典
    generator.load_state_dict(checkpoint)  # 将权重字典加载到模型中
    generator.eval()
        
    fake_datas = {}
    # times = 20
    for label, data in real_datas.items():
        fake_data = get_gen_data(label_dict,label_dim,data,label,generator,bins_data,sery_attrs,meta_attrs,batch_size)
        fake_datas[label] = fake_data
        for _ in range(expand_times - 1):
            fake_datas[label] += get_gen_data(label_dict,label_dim,data,label,generator,bins_data,sery_attrs,meta_attrs,batch_size)
            
        print("Generate data of", label)
    
    for label, data in fake_datas.items():
        save_data(label,data,meta_attrs,sery_attrs,result_folder_name)
        
