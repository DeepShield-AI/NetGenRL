import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, label_dim, seq_dim, max_seq_len, x_list, word_vec_dict, device):
        super(Discriminator, self).__init__()
        
        self.seq_dim = seq_dim
        self.label_dim = label_dim
        self.max_seq_len = max_seq_len
        self.x_list = x_list
        
        self.word_vec_list = [wv_tensor.to(device) for wv_tensor in word_vec_dict.values()]
        
        self.device = device
        self.lstm = nn.LSTM(input_size=seq_dim, hidden_size=512, num_layers=4, batch_first=True)
        self.lstm.flatten_parameters()
        
        self.length_fc = nn.Sequential(
            nn.Linear(max_seq_len, 128),
            nn.ReLU(True),
        )
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
        
        
    def forward(self, label, seq, length):
        self.lstm.flatten_parameters()
        
        length_one_hot = F.one_hot((length.clone().detach().long() - 1), num_classes=self.max_seq_len).float().to(self.device) # (B,max_seq_len)
        length_out = self.length_fc(length_one_hot) # (B,128)

        seq = seq.float() # (B, max_seq_len, seq_dim)

        
        packed_input = nn.utils.rnn.pack_padded_sequence(seq, length, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        # Unpack and get the last hidden state
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=seq.size(1)) # (B, max_seq_len, 512)

        len_expand = length_out.unsqueeze(1).expand(-1, output.size(1), -1) # (B,max_seq_len,128)
        combined = torch.cat([len_expand, output], dim=2)  # (B, max_seq_len, 512 + 128)
        
        hidden = self.fc(combined)  # (B, max_seq_len, 512)
        label_int = torch.argmax(label.clone(), 1) # (B)
            
        step_values = torch.stack([self.heads[idx](hidden[i]) for i, idx in enumerate(label_int)],dim=0) # (B,max_seq,1)

        return step_values
    
    def cal_value(self, lengths, validity):
        if validity.dim() == 3:
            validity = validity.squeeze(-1)
        max_T = validity.size(1) # seq_len
        length_devs = lengths.to(self.device)
        mask = (torch.arange(max_T, device=self.device)[None, :] < length_devs[:, None]).float()
        
        validity = validity * mask
        validity = validity.sum(dim=1)/length_devs
        return validity
    
    # label: (B,label_count), seq: (B,max_seq_len,seq_dim), length: (B)
    # def forward(self, label, seq, length):
    #     self.lstm.flatten_parameters()
        
    #     length_one_hot = F.one_hot((length.clone().detach().long() - 1), num_classes=self.max_seq_len).float().to(self.device) # (B,max_seq_len)
    #     length_out = self.length_fc(length_one_hot) # (B,128)
        
    #     seq = seq.float() # (B, max_seq_len, seq_dim)
    #     packed_input = nn.utils.rnn.pack_padded_sequence(seq, length, batch_first=True, enforce_sorted=False) 

    #     # Process with LSTM
    #     packed_output, (h_n, c_n) = self.lstm(packed_input)
        
    #     # Unpack and get the last hidden state
    #     output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=seq.size(1)) # (B, max_seq_len, 512)

    #     last_hidden_state = output[range(len(output)), (length - 1).to(torch.long), :]  # Get the last valid hidden state (B, 512)

    #     combined = torch.cat([length_out,last_hidden_state], dim=1) # (B, 128 + 512)
        
    #     hidden = self.fc(combined) # (B, 512)
             
    #     label_int = torch.argmax(label.clone(),1) #(B)
        
    #     validity = torch.stack([self.heads[idx](hidden[i]) for i, idx in enumerate(label_int)]) # (B,1)
        
    #     return validity
    
    

    
    def seq2wv(self, seq):
        seq_res = []
        for i in range(len(self.word_vec_list)):
            seq_one_hot = F.one_hot(seq[:,:,i].clone().detach().long(), num_classes=self.word_vec_list[i].shape[0]).float().to(self.device)
            seq_wv = torch.bmm(seq_one_hot,self.word_vec_list[i].unsqueeze(0).expand(seq_one_hot.size(0),-1,-1))
            seq_res.append(seq_wv)
        seq_res = torch.cat(seq_res, dim=2)
        return seq_res
