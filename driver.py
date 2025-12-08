import json
import os
from pre_process import pcap_process, ChiMerge, wordvec
from seqCGAN import model_train
from post_process import check_bestmodel, data_save, transformer_data_save
# from transformer import model_train

def check_and_make_forder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def construct_label_dict(pcap_folder, dataset):
    label_dict = {}
    count = 0
    for filename in os.listdir(f"./{pcap_folder}/{dataset}/"):
        if filename.endswith('.pcap') or filename.endswith('.pcapng'):
            label = filename.split('.')[0]
            label_dict[label] = count
            count += 1
    return label_dict
        

def driver():
    with open("./config.json","r") as file:
        config = json.load(file)
    print(config)
    
    label_dict = construct_label_dict(config['path']['pcap_folder'],config['path']['dataset'])
    sery_attr_names = list(config['attributes']['sery_attribute'].keys())
    port_attr_names = list(config['attributes']['port_attribute'].keys())
    ip_attr_names = list(config['attributes']['ip_attribute'].keys())
    param_dicts = dict(config['attributes']['sery_attribute'])
    param_dicts.update(dict(config['attributes']['port_attribute']))
    param_dicts.update(dict(config['attributes']['ip_attribute']))
    
    check_and_make_forder(config['path']['json_folder'])
    check_and_make_forder(f"./{config['path']['json_folder']}/{config['path']['dataset']}")
    check_and_make_forder(config['path']['bins_folder'])
    check_and_make_forder(config['path']['wordvec_folder'])
    check_and_make_forder(config['path']['model_folder'])
    check_and_make_forder(f"./{config['path']['model_folder']}/{config['path']['dataset']}")
    check_and_make_forder(config['path']['result_folder'])
    check_and_make_forder(f"./{config['path']['result_folder']}/{config['path']['dataset']}")
    
    # print("Processing pcap to json ...")
    
    # pcap_process.process_pcap(f"./{config['path']['pcap_folder']}/{config['path']['dataset']}",
    #                           label_dict,
    #                           f"./{config['path']['json_folder']}/{config['path']['dataset']}")
    
    print("Binning data ...")
    
    ChiMerge.chimerge(config['path']['dataset'],
                      config['path']['json_folder'],
                      config['path']['bins_folder'],
                      ip_attr_names,
                      port_attr_names,
                      sery_attr_names,
                      config['model_paras']['max_seq_len'],
                      param_dicts)
    
    # print("Pre-training wordvec model ...")
    
    # wordvec.run_word_vec(config['path']['dataset'],
    #                      config['path']['json_folder'],
    #                      config['path']['bins_folder'],
    #                      config['path']['wordvec_folder'],
    #                      ip_attr_names,
    #                      port_attr_names,
    #                      sery_attr_names,
    #                      config['model_paras']['max_seq_len'],
    #                      config['model_paras']['series_word_vec_size'],
    #                      config['model_paras']['meta_word_vec_size'])
    
    print("Training model ...")
    
    model_train.model_train(label_dict, 
                      config['path']['dataset'], 
                      config['path']['json_folder'],
                      config['path']['bins_folder'],
                      config['path']['model_folder'],
                      port_attr_names + ip_attr_names,
                      sery_attr_names,
                      config['model_paras'])

    # model_train.model_train(label_dict, 
    #                   config['path']['dataset'], 
    #                   config['path']['json_folder'],
    #                   config['path']['bins_folder'],
    #                   config['path']['model_folder'],
    #                   port_attr_names,
    #                   ip_attr_names,
    #                   sery_attr_names,
    #                   config['model_paras'])
    
    print("Choosing best model ...")
    model_id = check_bestmodel.check_models(label_dict,
                                            config['path']['dataset'],
                                            config['path']['json_folder'],
                                            config['path']['bins_folder'],
                                            config['path']['model_folder'],
                                            port_attr_names + ip_attr_names,
                                            sery_attr_names,
                                            config['model_paras']['batch_size'],
                                            config['model_paras']['max_seq_len'],
                                            config['model_paras']['checkpoint'],
                                            config['model_paras']['epoch'])
    
    
    print("Generating data ...")
    data_save.generate_data(label_dict,
                            config['path']['dataset'],
                            config['path']['json_folder'],
                            config['path']['bins_folder'],
                            config['path']['model_folder'],
                            config['path']['result_folder'],
                            port_attr_names + ip_attr_names,
                            sery_attr_names,
                            config['model_paras']['batch_size'],
                            config['model_paras']['max_seq_len'],
                            config['model_paras']['checkpoint'],
                            model_id,
                            config['model_paras']['expand_times'])
    
    # transformer_data_save.generate_data(label_dict,
    #                         config['path']['dataset'],
    #                         config['path']['json_folder'],
    #                         config['path']['bins_folder'],
    #                         config['path']['model_folder'],
    #                         config['path']['result_folder'],
    #                         port_attr_names,
    #                         ip_attr_names,
    #                         sery_attr_names,
    #                         config['model_paras']['batch_size'],
    #                         config['model_paras']['max_seq_len'],
    #                         config['model_paras']['expand_times'])
    

if __name__ == "__main__":
    driver()