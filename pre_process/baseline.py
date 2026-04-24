import numpy as np
import json
import scipy.stats as stats
import os
import math
from pre_process.ChiMerge import construct_data_dic


# =========================
# 🔧 工具函数（与原代码兼容）
# =========================
def flatten_data(data_dic, min_value, max_value):
    """
    将 {value: count} 转为展开后的数组（用于等频分箱）
    """
    values = []

    for data_list in data_dic.values():
        for value_dict in data_list:
            for k, v in value_dict.items():
                k = min(max(k, min_value), max_value)
                values.extend([k] * v)

    return np.array(values)


# =========================
# ✅ 方法1：等宽分箱
# =========================
def cal_bins_equal_width(data_dic, min_unit, min_value, max_value, num_bins=20):
    """
    输入输出格式完全兼容 cal_bins
    """
    width = (max_value - min_value) / num_bins

    intervals = []
    precision = max(0, -int(math.log10(min_unit)))

    for i in range(num_bins):
        left = min_value + i * width
        right = min_value + (i + 1) * width

        left = round(left, precision)
        right = round(right, precision)

        if left == right:
            continue

        intervals.append([left, right])

    return intervals


# =========================
# ✅ 方法2：等频分箱
# =========================
def cal_bins_equal_freq(data_dic, min_unit, min_value, max_value, num_bins=20):
    """
    等频(quantile)分箱
    """
    values = flatten_data(data_dic, min_value, max_value)

    if len(values) == 0:
        return []

    # 分位点
    quantiles = np.linspace(0, 1, num_bins + 1)
    bins = np.quantile(values, quantiles)

    # 去重（避免重复边界）
    bins = np.unique(bins)

    intervals = []
    precision = max(0, -int(math.log10(min_unit)))

    for i in range(len(bins) - 1):
        left = round(bins[i], precision)
        right = round(bins[i + 1], precision)

        if left == right:
            continue

        intervals.append([left, right])

    return intervals


def basemerge(dataset, json_folder, bins_folder, ip_attrs, port_attrs, sery_attrs, max_seq_len, params_dic):
    
    json_data_dic = {}
    for filename in os.listdir(f'./{json_folder}/{dataset}'):
        with open(f'./{json_folder}/{dataset}/{filename}', 'r') as f:
            json_data = json.load(f)
            json_data_dic[filename.split('.')[0]] = json_data

    attr_list = sery_attrs + port_attrs + ip_attrs
    # max_seq_len = 16

    data_dic = construct_data_dic(json_data_dic, attr_list, port_attrs, ip_attrs, sery_attrs, max_seq_len)
    # for dic in data_dic['pkt_len']['bittorrent']:
    #     print(dic)
    # # print(data_dic['pkt_len']['bittorrent'])
    # print(params_dic)
    # exit(0)

    np.set_printoptions(linewidth=2000)
    
    result_dic = {}
    
    bin_num_dic = {
        'time':100,
        'pkt_len': 500,
        'flags': 10,
        'ttl': 60,
        'src_port': 250,
        'dst_port': 250,
        'src_ip': 100,
        'dst_ip': 100,
    }
    
    for label in params_dic.keys():
        # if label != 'pkt_len':
        #     continue
        # merged_intervals,merged_data = cal_bins(data_dic[label],params_dic[label]['min_unit'],params_dic[label]['min_value'],params_dic[label]['max_value'])
        merged_intervals = cal_bins_equal_width(data_dic[label],params_dic[label]['min_unit'],params_dic[label]['min_value'],params_dic[label]['max_value'],bin_num_dic[label])
        # merged_intervals = cal_bins_equal_freq(data_dic[label],params_dic[label]['min_unit'],params_dic[label]['min_value'],params_dic[label]['max_value'])
        print(label,':', len(merged_intervals))
        for i in range(len(merged_intervals)):
            merged_intervals[i] = [round(merged_intervals[i][0],2),round(merged_intervals[i][1],2)]
            if i == 0:
                continue
            if merged_intervals[i][0] > round(merged_intervals[i-1][1] + params_dic[label]['min_unit'],2):
                merged_intervals[i][0] = round(merged_intervals[i-1][1] + params_dic[label]['min_unit'],2)
        # result_dic[label] = {'intervals':merged_intervals,'data':merged_data.tolist()}
        result_dic[label] = {'intervals':merged_intervals}
    
    json_str = json.dumps(result_dic)
    with open(f'./{bins_folder}/bins_{dataset}.json', 'w') as file:
        file.write(json_str)