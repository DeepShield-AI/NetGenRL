TOKEN_SEP = 256
TOKEN_EOS = 257
VOCAB_SIZE = 258 

class ByteTokenizer:
    def __init__(self):
        self.sep = TOKEN_SEP
        self.eos = TOKEN_EOS
        self.vocab_size = VOCAB_SIZE

    def bytes_to_ids(self, b: bytes):
        return [x for x in b]

    def ids_to_bytes(self, ids):
        bs = []
        for i in ids:
            if i == self.sep or i == self.eos:
                continue
            bs.append(i)
        return bytes(bs)