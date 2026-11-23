import argparse
import json
import os
from pre_process import ChiMerge, baseline
from stateFormer import model_train
from post_process import check_bestmodel, data_save

def check_and_make_forder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def construct_label_dict(pcap_folder, dataset, sort_labels=False):
    label_dict = {}
    count = 0
    filenames = os.listdir(f"./{pcap_folder}/{dataset}/")
    for filename in (sorted(filenames) if sort_labels else filenames):
        # if filename.endswith('.pcap') or filename.endswith('.pcapng'):
        #     label = filename.split('.')[0]
        #     label_dict[label] = count
        #     count += 1
        if filename.endswith('.json'):
            label = filename.split('.')[0]
            label_dict[label] = count
            count += 1
    return label_dict
        

def driver(config):
    method = config.get('binning', {}).get('method', 'chimerge')
    if method not in ('chimerge', 'equal_width', 'equal_freq'):
        raise ValueError(f"Unsupported binning method: {method}")
    if config.get('experiment', {}).get('fresh_start', False):
        from pathlib import Path
        paths = config['path']
        targets = [Path(paths['bins_folder']) / f"bins_{paths['dataset']}.json",
                   Path(paths['model_folder']) / paths['dataset'],
                   Path(paths['result_folder']) / paths['dataset']]
        for target in targets:
            if target.exists():
                raise FileExistsError(f"Fresh experiment refuses existing output: {target}")
    checkpoint = config['model_paras']['checkpoint']
    epochs = config['model_paras']['epoch']
    if checkpoint <= 0 or epochs <= 0 or epochs % checkpoint:
        raise ValueError("epoch must be positive and divisible by checkpoint")
    seed = config.get('experiment', {}).get('seed')
    if seed is not None:
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    label_dict = construct_label_dict(config['path']['json_folder'], config['path']['dataset'],
                                      sort_labels=seed is not None)
    sery_attr_names = list(config['attributes']['sery_attribute'].keys())
    port_attr_names = list(config['attributes']['port_attribute'].keys())
    ip_attr_names = list(config['attributes']['ip_attribute'].keys())
    param_dicts = dict(config['attributes']['sery_attribute'])
    param_dicts.update(dict(config['attributes']['port_attribute']))
    param_dicts.update(dict(config['attributes']['ip_attribute']))
    
    check_and_make_forder(config['path']['json_folder'])
    check_and_make_forder(f"./{config['path']['json_folder']}/{config['path']['dataset']}")
    check_and_make_forder(config['path']['bins_folder'])
    check_and_make_forder(config['path']['model_folder'])
    check_and_make_forder(f"./{config['path']['model_folder']}/{config['path']['dataset']}")
    check_and_make_forder(config['path']['result_folder'])
    check_and_make_forder(f"./{config['path']['result_folder']}/{config['path']['dataset']}")
    
    print(f"Binning data: {method} ...")
    bin_args = (config['path']['dataset'], config['path']['json_folder'],
                config['path']['bins_folder'], ip_attr_names, port_attr_names,
                sery_attr_names, config['model_paras']['max_seq_len'], param_dicts)
    if method == 'chimerge':
        ChiMerge.chimerge(*bin_args)
    else:
        baseline.basemerge(*bin_args, method=method,
                           bin_counts=config['binning'].get('bin_counts'))

    print("Training model ...")
    
    model_train.model_train(label_dict, 
                      config['path']['dataset'], 
                      config['path']['json_folder'],
                      config['path']['bins_folder'],
                      config['path']['model_folder'],
                      port_attr_names + ip_attr_names,
                      sery_attr_names,
                      config['model_paras'])
    
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

def read_config(path="./config.json"):
    with open(path, "r") as file:
        return json.load(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binning, training, selection and generation")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--binning-method", choices=("chimerge", "equal_width", "equal_freq"))
    args = parser.parse_args()
    config = read_config(args.config)
    if args.binning_method:
        config.setdefault('binning', {})['method'] = args.binning_method
    driver(config)
