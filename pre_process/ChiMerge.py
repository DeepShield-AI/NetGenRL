# %%
import numpy as np
import json
import scipy.stats as stats
import os
import math


# %%
def threshold_merge(intervals, data, min_unit, threshold = 1):
    new_intervals = []
    new_data = np.empty((data.shape[0],0))
    
    flag = False
    for i in range(len(intervals)):
        if flag and np.max(data[:,i]) <= threshold:
            new_intervals[-1][1] = intervals[i][1]
            new_data[:,-1] = new_data[:,-1] + data[:,i]
            continue
        if not flag and np.max(data[:,i]) <= threshold:
            new_intervals.append(intervals[i])
            if i > 0:
                new_intervals[-1][0] = new_intervals[-2][0] + min_unit
            new_data = np.hstack((new_data,data[:,i].reshape(-1, 1)))
            flag = True
            continue
        if flag and np.max(data[:,i]) > threshold:
            new_intervals[-1][1] = intervals[i][0] - min_unit
            new_intervals.append(intervals[i])
            new_data = np.hstack((new_data,data[:,i].reshape(-1, 1)))
            flag = False
            continue
        new_intervals.append(intervals[i])
        new_data = np.hstack((new_data,data[:,i].reshape(-1, 1)))
        flag = False

    return new_intervals,new_data

# %%
def chi_cal(interval_l,interval_r,num_ls,num_rs):
    left_board = interval_l[0]
    right_board = interval_r[1]
    empty_len = interval_r[0] - interval_l[1] - 1
    left_len = interval_l[1] - interval_l[0] + 1
    right_len = interval_r[1] - interval_r[0] + 1
    totals = num_ls + num_rs
    length = right_board - left_board + 1
    ps = totals/length
    chi_2 = empty_len * ps + np.where(ps != 0, ((num_ls / left_len - ps) ** 2 / ps * left_len + (num_rs / right_len - ps) ** 2 / ps * right_len), 0)
    return np.sum(chi_2)

# %%
def chi_merge(intervals, data, chi_threshold=3.84):
    m, _ = data.shape
    
    if len(intervals) == 0:
        return intervals,data

    chi2_vals = np.zeros(len(intervals) - 1)

    if len(chi2_vals) == 0:
        return intervals,data

    for i in range(len(intervals)-1):
        chi2_vals[i] = chi_cal(intervals[i],intervals[i+1],data[:,i],data[:,i+1])

    min_chi2_id = 0

    while True:
        if len(chi2_vals) == 0:
            return intervals,data
        min_chi2_id = np.argmin(chi2_vals)
        
        if chi2_vals[min_chi2_id] >= chi_threshold:
            break

        intervals[min_chi2_id][1] = intervals[min_chi2_id+1][1]
        intervals.pop(min_chi2_id+1)

        data[:,min_chi2_id] = data[:,min_chi2_id] + data[:,min_chi2_id+1]
        data = np.delete(data,min_chi2_id+1,axis=1)

        chi2_vals = np.delete(chi2_vals,[min_chi2_id])
        
        if min_chi2_id != 0:
            chi2_vals[min_chi2_id - 1] = chi_cal(intervals[min_chi2_id-1],intervals[min_chi2_id],data[:,min_chi2_id -1],data[:,min_chi2_id])
        if min_chi2_id < len(chi2_vals):
            chi2_vals[min_chi2_id] = chi_cal(intervals[min_chi2_id],intervals[min_chi2_id+1],data[:,min_chi2_id],data[:,min_chi2_id+1])
    return intervals,data

# %%
def construct_data_dic(json_data_dic, attribute_list, port_attr_list, ip_attr_list, series_attr_list, max_seq_len):
    data_dic = {attribute: {} for attribute in attribute_list}

    for label, json_data in json_data_dic.items():
        for attribute in attribute_list:
            data_dic[attribute][label] = []
        for item in json_data:
            for ip_attr in ip_attr_list:
                data_dic[ip_attr][label].append({})
                if item[ip_attr] not in data_dic[ip_attr][label][0]:
                    data_dic[ip_attr][label][0][item[ip_attr]] = 0
                data_dic[ip_attr][label][0][item[ip_attr]] += 1
            for port_attr in port_attr_list:
                data_dic[port_attr][label].append({})
                if item[port_attr] not in data_dic[port_attr][label][0]:
                    data_dic[port_attr][label][0][item[port_attr]] = 0
                data_dic[port_attr][label][0][item[port_attr]] += 1
            
            for i,pkt in enumerate(item['series']):
                if i >= max_seq_len:
                    break
                # print(pkt)
                # exit(0)
                for sery_attr in series_attr_list:
                    if len(data_dic[sery_attr][label]) <= i:
                        data_dic[sery_attr][label].append({})
                    if pkt[sery_attr] not in data_dic[sery_attr][label][i]:
                        data_dic[sery_attr][label][i][pkt[sery_attr]] = 0
                    data_dic[sery_attr][label][i][pkt[sery_attr]] += 1
                    

    return data_dic
    

# %%
def construct_intervals_data(data_dic, min_value, max_value, min_unit):

    # labels = data_dic.keys()
    # dics = list(data_dic.values())
    seq_len = len(data_dic)

    intervals = []
    data_list = []

    tmp_dict = {}

    id = 0
    precision = max(0, -int(math.log10(min_unit)))
    # for value in dics:
    for value in data_dic:
        for k, v in value.items():
            k = min(max(k,min_value),max_value)
            k = round(k,precision)
            if k not in tmp_dict:
                # tmp_dict[k] = [0 for _ in range(len(labels))]
                tmp_dict[k] = [0 for _ in range(seq_len)]
            tmp_dict[k][id] = v
        id += 1

    tmp_lists = sorted(tmp_dict.items(), key=lambda x: x[0])

    for (iter, li) in tmp_lists:
        intervals.append([iter,iter])
        data_list.append(li)

    data = np.array(data_list)

    return intervals,data.transpose()

# %%

def fill_intervals(intervals, data, min_unit, min_value, max_value):
    now_value = min_value
    for i, interval in enumerate(intervals):
        if now_value + min_unit < interval[0]:
            intervals.insert(i,[now_value,interval[0] - min_unit])
            data = np.insert(data,i,0,axis=1)
        now_value = interval[1] + min_unit
    if now_value + min_unit < max_value:
        intervals.append([now_value, max_value])
        data = np.insert(data,len(intervals)-1,0,axis=1)
    return intervals,data

def cal_bins_one_category(data_list, min_unit, min_value, max_value):
    intervals, data = construct_intervals_data(data_list, min_value, max_value, min_unit)
    # print(data)
    new_intervals,new_data = threshold_merge(intervals,data, min_unit)

    alpha = 0.01
    df = new_data.shape[0] - 1
    critical_value = stats.chi2.ppf(1 - alpha, df)

    merged_intervals,merged_data = chi_merge(new_intervals,new_data,chi_threshold=critical_value)
    return merged_intervals,merged_data

def integrate_bins(intervals_dict, min_unit, min_value):
    integrated_intervals = []
    
    intervals = []
    for itv in intervals_dict.values():
        intervals += itv
    sorted_intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    # print(sorted_intervals)
    
    left_boards = [itv[0] for itv in sorted_intervals]
    right_boards = sorted([itv[1] for itv in sorted_intervals])
    # print(left_boards)
    # print(right_boards)
    # left_count = 0
    right_id = 0
    last_value = min_value - min_unit
    for i in range(len(left_boards)):
        if i == len(left_boards) - 1:
            # print(left_count)
            while right_id < len(right_boards):
                # left_count -= 1
                if right_boards[right_id] < left_boards[i]:
                    right_id += 1
                    continue
                integrated_intervals.append([left_boards[i],right_boards[right_id]])
                # print("a",i,right_id,integrated_intervals[-1])
                last_value = right_boards[right_id]
                left_boards[i] = right_boards[right_id] + min_unit
                right_id += 1
            break
        # if right_boards[right_id] == left_boards[i]:
        #     left_count -= 1
        #     right_id += 1
        #     if left_boards[i] <= last_value:
        #         continue
        #     integrated_intervals.append([left_boards[i],left_boards[i]])
        #     last_value = left_boards[i]
        #     continue
        if left_boards[i] == left_boards[i+1]:
            continue
        
        while left_boards[i+1] > right_boards[right_id]:
            # if left_count == 0:
            #     print("error")
            # left_count -= 1
            if i < right_id:
                print("error")
                break
            if right_boards[right_id] < left_boards[i]:
                right_id += 1
                continue
            integrated_intervals.append([left_boards[i],right_boards[right_id]])
            # print("c",i,right_id,integrated_intervals[-1])
            last_value = right_boards[right_id]
            left_boards[i] = right_boards[right_id] + min_unit
            right_id += 1
            
        if i < right_id:
            # print(i)
            continue
            
        # if left_boards[i+1] <= right_boards[right_id]:
        if left_boards[i+1] - min_unit < last_value + min_unit:
            continue
        # print(last_value)
        integrated_intervals.append([left_boards[i],left_boards[i+1]-min_unit])
        # print("b",i,right_id,integrated_intervals[-1])
        last_value = left_boards[i+1] - min_unit
        
    
    # for i,itv in enumerate(integrated_intervals):
    #     if round(itv[0],2) > round(itv[1],2):
    #         print("error 1:",itv)
    #     if i == len(integrated_intervals) - 1:
    #         continue
    #     if round(itv[1],2) > round(integrated_intervals[i+1][0] - min_unit,2):
    #         print("error 2:",itv,integrated_intervals[i+1])
    
    # print(integrated_intervals)
    # exit(0)
    
    return integrated_intervals

def filter_bins(intervals,data_dic,min_unit,min_value,max_value):
    filtered_intervals = []
    data_array = []
    precision = max(0, -int(math.log10(min_unit)))
    for data_list in data_dic.values():
        for value in data_list:
            for k in value.keys():
                k = min(max(k,min_value),max_value)
                # k = round(k / min_unit) * min_unit
                k = round(k,precision)
                if k not in data_array:
                    data_array.append(k)
    data_array = sorted(data_array)
    
    id = 0
    for itv in intervals:
        while id < len(data_array) and itv[0] > data_array[id]:
            # print(itv,data_array[id])
            id += 1
        if itv[1] < data_array[id]:
            # print(itv,data_array[id-1],data_array[id])
            continue 
        filtered_intervals.append(itv)
    return filtered_intervals

def cal_bins(data_dic, min_unit, min_value, max_value):
    
    intervals_dict = {}
    for category, data_list in data_dic.items():
        intervals, _ = cal_bins_one_category(data_list,min_unit,min_value,max_value)
        precision = max(0, -int(math.log10(min_unit)))
        for i in range(len(intervals)):
            # intervals[i] = [round(intervals[i][0],2),round(intervals[i][1],2)]
            intervals[i] = [round(intervals[i][0],precision),round(intervals[i][1],precision)]
        intervals_dict[category] = intervals
    integrated_intervals = integrate_bins(intervals_dict,min_unit,min_value)
    filtered_intervals = filter_bins(integrated_intervals,data_dic,min_unit,min_value,max_value)
    return filtered_intervals
    
    

# %%
def chimerge(dataset, json_folder, bins_folder, ip_attrs, port_attrs, sery_attrs, max_seq_len, params_dic):
    
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
    
    for label in params_dic.keys():
        # if label != 'pkt_len':
        #     continue
        # merged_intervals,merged_data = cal_bins(data_dic[label],params_dic[label]['min_unit'],params_dic[label]['min_value'],params_dic[label]['max_value'])
        merged_intervals = cal_bins(data_dic[label],params_dic[label]['min_unit'],params_dic[label]['min_value'],params_dic[label]['max_value'])
        print(label,':', len(merged_intervals))
        for i in range(len(merged_intervals)):
            merged_intervals[i] = [round(merged_intervals[i][0],2),round(merged_intervals[i][1],2)]
        # result_dic[label] = {'intervals':merged_intervals,'data':merged_data.tolist()}
        result_dic[label] = {'intervals':merged_intervals}
    
    json_str = json.dumps(result_dic)
    with open(f'./{bins_folder}/bins_{dataset}.json', 'w') as file:
        file.write(json_str)

    