# config.py

# MAX_SEQ_LEN = 512
EMBED_DIM = 256
LABEL_EMB_DIM = 64
LEN_EMB_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 6
FFN_DIM = 1024
DROPOUT = 0.05
# BATCH_SIZE = 8
LR = 5e-5
WD = 1e-4
TEMPERATURE = 1.0

# SEPS = {"LABEL_SEP":0, "PREFIX_SEP":1, "PACKET_SEP":2, "SUFFIX_SEP":3, "EOS_SEP":4}
SEPS = {"SEP":0}

def cal_max_len(port_attrs,max_pkt_len,sery_attrs,ip_attrs):
    # return 3 + len(port_attrs) + 1 + max_pkt_len * (len(sery_attrs) + 1) + 1 + len(ip_attrs) + 1
    return 3 + max_pkt_len * (8 + 1 + 1 + 1 + 1) + 1 + len(port_attrs)*2 + len(ip_attrs)*4 + 1
# LABELS = {'skype':0,'spotify':1,'voipbuster':2, 'youtube':3}

# DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
