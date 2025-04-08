# %%
import os
from collections import defaultdict
from scapy.all import rdpcap, IP, TCP, UDP
import csv
import json
import ipaddress

# %%
def count_flows(pcap_file):
    packet_count = 0
    flows = defaultdict(int)  # 存储流的字典，以（源IP, 源端口, 目标IP, 目标端口）为键
    
    # 使用 scapy 读取 pcap 文件
    packets = rdpcap(pcap_file)
    
    for packet in packets:
        packet_count += 1
        
        # 确保包中有 IP 和 TCP 或 UDP 层
        if IP in packet:
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            if TCP in packet:
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
            elif UDP in packet:
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport
            else:
                continue
            
            # 创建流的标识（源IP, 源端口, 目标IP, 目标端口）
            flow_1 = (src_ip, src_port, dst_ip, dst_port)
            flow_2 = (dst_ip, dst_port, src_ip, src_port)  # 反向流
            
            # 记录流
            flows[flow_1] += 1
            flows[flow_2] += 1
    
    return packet_count, len(flows)/2

# %%
def process_flows(pcap_file):
    flows = {}
    
    # 使用 scapy 读取 pcap 文件
    packets = rdpcap(pcap_file)
    
    for packet in packets:
        # packet_count += 1
        
        # 确保包中有 IP 和 TCP 或 UDP 层
        if IP in packet:
            src_ip = int(ipaddress.ip_address(packet[IP].src))
            dst_ip = int(ipaddress.ip_address(packet[IP].dst))
            if TCP in packet:
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
                protocol = 6
            elif UDP in packet:
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport
                protocol = 17
            else:
                continue

            pkt = {}
            pkt['time'] = float(packet.time)
            pkt['pkt_len'] = len(packet)
            pkt['flags'] = int(packet[TCP].flags) if TCP in packet else 0
            pkt['ttl'] = packet[IP].ttl
            
            # 创建流的标识（源IP, 源端口, 目标IP, 目标端口）
            flow_1 = (src_ip, src_port, dst_ip, dst_port, protocol)
            flow_2 = (dst_ip, dst_port, src_ip, src_port, protocol)  # 反向流

            if flow_1 in flows.keys():
                pkt['time'] -= flows[flow_1]['last_time']
                flows[flow_1]['last_time'] = float(packet.time)
                flows[flow_1]['series'].append(pkt)
                continue
            if flow_2 in flows.keys():
                pkt['time'] -= flows[flow_2]['last_time']
                flows[flow_2]['last_time'] = float(packet.time)
                pkt['pkt_len'] = -pkt['pkt_len']
                flows[flow_2]['series'].append(pkt)
                continue

            flows[flow_1] = {}
            flows[flow_1]['last_time'] = float(packet.time)
            pkt['time'] = 0
            flows[flow_1]['series'] = []
            flows[flow_1]['series'].append(pkt)
            flows[flow_1]['src_ip'] = src_ip 
            flows[flow_1]['src_port'] = src_port
            flows[flow_1]['dst_ip'] = dst_ip
            flows[flow_1]['dst_port'] = dst_port
            flows[flow_1]['protocol'] = protocol
    
    return list(flows.values())

# %%
def process_pcap_folder(folder_path):
    total_packets = 0
    total_flows = 0
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.pcap') or filename.endswith('.pcapng'):
            pcap_file = os.path.join(folder_path, filename)
            packet_count, flow_count = count_flows(pcap_file)
            total_packets += packet_count
            total_flows += flow_count
            print(f"File: {filename} - Packets: {packet_count} - Flows: {flow_count}")
    
    print("\nTotal Packets:", total_packets)
    print("Total Flows:", total_flows)

# %%
def save_flows_to_json(flows, output_file):
    with open(output_file, 'w') as file:
        json.dump(flows, file)  

# %%
def process_pcap(folder_path_list, label_str):
    
    flows = []
    for folder_path in folder_path_list:
        for filename in os.listdir(folder_path):
            if filename.endswith('.pcap') or filename.endswith('.pcapng'):
                if label_str not in filename:
                    continue
                pcap_file = os.path.join(folder_path, filename)
            
                flows += (process_flows(pcap_file))
                print(f"File: {filename}")
            
    save_flows_to_json(flows, './Data-json/'+label_str+'.json')
    
    print(f"Total Flows of {label_str}:", len(flows))

# %%
folder_path_list = ['data']
LABEL_DICT = {'benign_http': 0, 'benign_rtp': 1, 'bruteforce_dns': 2, 'bruteforce_ssh': 3, 'ddos_ack': 4, 'ddos_dns': 5, 'ddos_http': 6, 'ddos_syn': 7, 'scan_bruteforce': 8}
for label_str in LABEL_DICT.keys():
    process_pcap(folder_path_list, label_str)


