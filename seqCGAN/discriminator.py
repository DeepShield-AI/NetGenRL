import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, label_dim, seq_dim, max_seq_len, x_list, word_vec_dict, device):
        super(Discriminator, self).__init__()
        
        self.seq_dim = seq_dim
        # self.metadata_dim = metadata_dim
        self.label_dim = label_dim
        self.max_seq_len = max_seq_len
        self.x_list = x_list
        # self.word_vec_dict = word_vec_dict
        
        self.word_vec_list = [wv_tensor.to(device) for wv_tensor in word_vec_dict.values()]
        
        self.device = device
        
        # self.emb = nn.ModuleList([
        #     nn.Embedding(x_len, 128) for x_len in x_list
        # ]) 

        # self.lstm = nn.LSTM(input_size=128 * len(x_list), hidden_size=512, num_layers=4, batch_first=True)
        self.lstm = nn.LSTM(input_size=seq_dim, hidden_size=512, num_layers=4, batch_first=True)
        self.lstm.flatten_parameters()
        
        self.length_fc = nn.Sequential(
            nn.Linear(max_seq_len, 128),
            nn.ReLU(True),
        )
        
        # 合并图像特征、标签特征和元数据特征
        self.fc = nn.Sequential(
            nn.Linear(128+512, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 1),
                nn.Identity()
            ) for _ in range(self.label_dim)
        ])
        
        # self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, label, seq, length):
        self.lstm.flatten_parameters()
        
        length_one_hot = F.one_hot((length.clone().detach().long() - 1), num_classes=self.max_seq_len).float().to(self.device)
        
        # 处理标签
        length_out = self.length_fc(length_one_hot)
        
        seq = seq.float()
        # seq = torch.stack([seq[:,:,i]/x_len for i, x_len in enumerate(self.x_list)], dim=2)
        
        # seq_emb = torch.cat([self.emb[i](seq[:,:,i]) for i in range(len(self.x_list))], dim=2)
        
        # packed_input = nn.utils.rnn.pack_padded_sequence(seq_emb, length, batch_first=True, enforce_sorted=False)
        packed_input = nn.utils.rnn.pack_padded_sequence(seq, length, batch_first=True, enforce_sorted=False)

        # Process with LSTM
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        
        # Unpack and get the last hidden state
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        last_hidden_state = output[range(len(output)), (length - 1).to(torch.long), :]  # Get the last valid hidden state

        combined = torch.cat([length_out,last_hidden_state], dim=1)
        
        hidden = self.fc(combined)
             
        label_int = torch.argmax(label.clone(),1)
        
        validity_outputs = torch.stack([self.heads[idx](hidden) for idx in range(len(self.heads))],dim=1)
        # # print(validity_outputs.shape)
        
        indices = label_int.view(-1, 1, 1).expand(-1, -1, validity_outputs.size(2))
        # # print(indices.shape)

        validity = torch.gather(validity_outputs, dim=1, index=indices).squeeze(1)
        # print(validity.shape)
        
        # 判别真假
        # validity = torch.stack([self.heads[idx](hidden[i]) for i, idx in enumerate(label_int)])
        
        # prob = self.softmax(validity)
        # return prob
        return validity
    
    def seq2wv(self, seq):
        seq_res = []
        for i in range(len(self.word_vec_list)):
            seq_one_hot = F.one_hot(seq[:,:,i].clone().detach().long(), num_classes=self.word_vec_list[i].shape[0]).float().to(self.device)
            seq_wv = torch.bmm(seq_one_hot,self.word_vec_list[i].unsqueeze(0).expand(seq_one_hot.size(0),-1,-1))
            seq_res.append(seq_wv)
        seq_res = torch.cat(seq_res, dim=2)
        # print(seq_res.shape)
        return seq_res
