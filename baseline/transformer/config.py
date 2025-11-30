# config.py

MAX_SEQ_LEN = 512
EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
FFN_DIM = 1024
DROPOUT = 0.2
BATCH_SIZE = 8
LR = 1e-4
WD = 0.01
LABELS = {'skype':0,'spotify':1,'voipbuster':2, 'youtube':3}

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
