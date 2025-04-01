import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import *

class Generator(nn.Module):
    def __init__(self, label_dim, seq_dim, max_seq_len, x_list, device):
        super(Generator, self).__init__()
        
        self.label_dim = label_dim
        # self.noise_dim = noise_dim
        self.hidden_dim = 512
        self.lstm_layers = 4
        self.embedding_dim = 128
        self.attribute_dim = 64
        self.max_seq_len = max_seq_len
        self.seq_dim = seq_dim
        self.pred_dim = sum(x_list)
        self.x_list = x_list # list of seq value dim
        self.device = device
        self.count_list = []
        # count = 0
        # for x_len in x_list:
        #     count += x_len
        #     self.count_list.append(count)
                
        self.emb = nn.ModuleList([
            nn.Embedding(x_len, self.embedding_dim) for x_len in x_list
        ]) 
        
        # self.label_fc = nn.Sequential(
        #     nn.Linear(label_dim, 64),
        #     nn.ReLU(True),
        # )
        
        # self.emb_now = nn.ModuleList([
        #     nn.Embedding(x_len, self.embedding_dim) for x_len in x_list[:-1]
        # ]) 
        
        self.pad_embedding = nn.Parameter(torch.zeros(1, self.attribute_dim))
        
        self.condition_fix = nn.ModuleList([
            nn.Sequential(
                nn.Linear((i + 1)*self.embedding_dim, self.attribute_dim),
                nn.ReLU(True),
            ) for i in range(len(x_list)-1)
        ])
        
        self.length_fc = nn.Sequential(
            nn.Linear(max_seq_len, 64),
            nn.ReLU(True),
        ) 
        
        # self.combine_fc = nn.Sequential(
        #     nn.Linear(128*len(x_list)+64+64,self.hidden_dim),
        #     nn.ReLU(True)
        # )
        
        self.combine_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128*len(x_list)+64,self.hidden_dim),
                nn.ReLU(True)
            ) for _ in range(self.label_dim)
        ])
        
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.lstm_layers, batch_first=True)
        self.lstm.flatten_parameters()

        self.output_layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim + self.attribute_dim, x_len),
                nn.Identity()
            ) for x_len in x_list
        ])
        # self.output_layer = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(self.hidden_dim, x_len),
        #         nn.Identity()
        #     ) for x_len in x_list
        # ])
        # self.output_layer = nn.Sequential(
        #     nn.Linear(self.hidden_dim, self.pred_dim),
        #     nn.Identity()
        # )

        
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def init_hidden(self, batch_size):
        h = torch.zeros((self.lstm_layers, batch_size, self.hidden_dim),device=self.device)
        c = torch.zeros((self.lstm_layers, batch_size, self.hidden_dim),device=self.device)
        return h, c
        
    def forward(self, label, length, x, x_now): # x: (batch_size, seq_len, seq_dim)
        # torch.cuda.empty_cache()
        self.lstm.flatten_parameters()
        length_one_hot = F.one_hot((length.clone().detach().long() - 1), num_classes=self.max_seq_len).float().to(self.device)
        
        # label_out = self.label_fc(label)
        label_int = torch.argmax(label.clone(),1)
        # length_out = torch.stack([self.length_fc[idx](length_one_hot[i]) for i, idx in enumerate(label_int)])
        length_out = self.length_fc(length_one_hot)
        
        # emb_list = torch.stack([torch.cat([self.emb[idx][j](x[i,:,j]) for j in range(len(self.x_list))], dim=1) for i, idx in enumerate(label_int)],dim=0)
        emb_list = torch.cat([self.emb[i](x[:,:,i]) for i in range(len(self.x_list))], dim=2)
        
        now_emb_list = torch.cat([self.emb[i](x_now[:,:,i]) for i in range(len(self.x_list)-1)], dim=2)
        attribute_features = [self.condition_fix[i](now_emb_list[:,:,:(i+1)*self.embedding_dim]) for i in range(len(self.x_list) - 1)]
        # attribute_features = [torch.zeros(x_now.shape[0],x_now.shape[1],self.attribute_dim).to(self.device)] + attribute_features
        attribute_features = [self.pad_embedding.expand(x_now.shape[0],x_now.shape[1],self.attribute_dim)] + attribute_features
        
        # label_expand = label_out.unsqueeze(1).expand(-1, emb_list.size(1), -1)
        length_expand = length_out.unsqueeze(1).expand(-1, emb_list.size(1), -1)

        combined = torch.cat([emb_list, length_expand], dim=2)
        # print(combined.shape)
        # combined_out = self.combine_fc(combined)   
        # combined_out = torch.stack([self.combine_fc[idx](combined[i,:,:]) for i, idx in enumerate(label_int)],dim = 0)
        
        fc_outputs = torch.stack([self.combine_fc[idx](combined) for idx in range(len(self.combine_fc))], dim=1)
        # print(fc_outputs.shape)
        
        indices = label_int.view(-1, 1, 1, 1).expand(-1, -1, fc_outputs.size(2), fc_outputs.size(3))
        # print(indices.shape)

        combined_out = torch.gather(fc_outputs, dim=1, index=indices).squeeze(1)

        # print(combined_out.shape)
        h0, c0 = self.init_hidden(combined_out.size(0))
        hidden_output, (h, c) = self.lstm(combined_out, (h0, c0))
        
        
        # output = self.output_layer(hidden_output.contiguous().view(-1, self.hidden_dim))
        
        # count = 0
        preds = []
        for i in range(len(self.x_list)):
            hidden_w_attr = torch.cat([hidden_output, attribute_features[i]], dim=2)
            output = self.output_layer[i](hidden_w_attr.contiguous().view(-1, self.hidden_dim + self.attribute_dim))
            # pred = self.softmax(output[:,count:count + x_len])
            # output = self.output_layer[i](hidden_output.contiguous().view(-1, self.hidden_dim))
            pred = self.softmax(output)
            preds.append(pred)
            # count += x_len
        
        # count = 0  
        # for x_len in self.x_list:
        #     output = self.output_layer(hidden_output.contiguous().view(-1, self.hidden_dim))
        #     pred = self.softmax(output[:,count:count + x_len])
        #     preds.append(pred)
        #     count += x_len
            
        preds = torch.cat(preds, dim=1).view(-1, self.max_seq_len, self.pred_dim) 
        return preds
    
    def get_LSTM_hidden(self, label, length, x, h, c):
        
        length_one_hot = F.one_hot((length.clone().detach().long() - 1), num_classes=self.max_seq_len).float().to(self.device)
        
        length_out = self.length_fc(length_one_hot)
        label_int = torch.argmax(label.clone(),1)
        
        emb_list = torch.cat([self.emb[i](x[:,i]) for i in range(len(self.x_list))], dim=1)
        
        combined = torch.cat([emb_list, length_out], dim=1)

        fc_outputs = torch.stack([self.combine_fc[idx](combined) for idx in range(len(self.combine_fc))], dim=1)
        
        indices = label_int.view(-1, 1, 1).expand(-1, -1, fc_outputs.size(2))

        combined_out = torch.gather(fc_outputs, dim=1, index=indices)

        hidden_output, (h, c) = self.lstm(combined_out, (h, c))
        
        hidden_output = hidden_output.squeeze(1)        
            
        return hidden_output, h, c
    
    def step_w_hidden(self, hidden_output, x_now, dim_now):
        if dim_now > 0 and x_now is not None:
            now_emb_list = torch.cat([self.emb[i](x_now[:,i]) for i in range(dim_now)], dim=1)
            attribute_feature = self.condition_fix[dim_now - 1](now_emb_list)
        else:
            # attribute_feature = torch.zeros(x.shape[0],self.attribute_dim).to(self.device)
            attribute_feature = self.pad_embedding.expand(hidden_output.shape[0],self.attribute_dim)
            
        hidden_w_attr = torch.cat([hidden_output, attribute_feature], dim=1)
        output = self.output_layer[dim_now](hidden_w_attr.contiguous())
        # output = self.output_layer[dim_now](hidden_output.view(-1, self.hidden_dim))
        # output = self.output_layer(hidden_output.view(-1, self.hidden_dim))
        # count = sum(self.x_list[:dim_now])
        pred = F.softmax(output,dim=-1)
        return pred
    
    def step(self, label, length, x, x_now, dim_now, h, c):
        """
        Args:
            x: (batch_size, seq_dim), sequence of tokens generated by generator
            h: (1, batch_size, hidden_dim), lstm hidden state
            c: (1, batch_size, hidden_dim), lstm cell state
        """
        # torch.cuda.empty_cache()
        # self.lstm.flatten_parameters()
        length_one_hot = F.one_hot((length.clone().detach().long() - 1), num_classes=self.max_seq_len).float().to(self.device)
        
        # label_out = self.label_fc(label)
        length_out = self.length_fc(length_one_hot)
        label_int = torch.argmax(label.clone(),1)
        # length_out = torch.stack([self.length_fc[idx](length_one_hot[i]) for i, idx in enumerate(label_int)])
        
        # print(x.shape)
        emb_list = torch.cat([self.emb[i](x[:,i]) for i in range(len(self.x_list))], dim=1)
        
        if dim_now > 0 and x_now is not None:
            now_emb_list = torch.cat([self.emb[i](x_now[:,i]) for i in range(dim_now)], dim=1)
            attribute_feature = self.condition_fix[dim_now - 1](now_emb_list)
        else:
            # attribute_feature = torch.zeros(x.shape[0],self.attribute_dim).to(self.device)
            attribute_feature = self.pad_embedding.expand(x.shape[0],self.attribute_dim)

        # combined = torch.cat([emb_list, label_out, length_out], dim=1)
        # combined_out = self.combine_fc(combined)   
        combined = torch.cat([emb_list, length_out], dim=1)
        # combined_out = torch.stack([self.combine_fc[idx](combined[i]) for i, idx in enumerate(label_int)])

        fc_outputs = torch.stack([self.combine_fc[idx](combined) for idx in range(len(self.combine_fc))], dim=1)
        
        indices = label_int.view(-1, 1, 1).expand(-1, -1, fc_outputs.size(2))

        combined_out = torch.gather(fc_outputs, dim=1, index=indices)

        hidden_output, (h, c) = self.lstm(combined_out, (h, c))
        
        hidden_output = hidden_output.squeeze(1)
        # print(hidden_output.shape, attribute_feature.shape)
        # hidden_w_attr = torch.cat([hidden_output, attribute_feature], dim=1)
        # output = self.output_layer[dim_now](hidden_w_attr.contiguous())
        output = self.output_layer[dim_now](hidden_output.view(-1, self.hidden_dim))
        pred = F.softmax(output,dim=-1)
        
        # output = self.output_layer(hidden_output.view(-1, self.hidden_dim))
        
        # count = 0
        # preds = []
        # for x_len in self.x_list:
        #     pred = F.softmax(output[:,count:count + x_len],dim=-1)
        #     preds.append(pred)
        #     count += x_len
            
        # preds = torch.cat(preds, dim=1) 
        
        
            
        return pred, h, c
    
    
    def sample(self, batch_size, label, length, x=None):
        flag = False # whether sample from zero
        if x is None:
            flag = True
        if flag:
            x = torch.zeros((batch_size, self.seq_dim)).long().to(self.device)
            
        h, c = self.init_hidden(batch_size)
        samples = []
        if flag:
            first = []
            for i in range(self.max_seq_len): 
                last = None
                sam = []
                hidden_output, h, c = self.get_LSTM_hidden(label, length, x, h, c)
                if i == 0:
                    for j in range(len(self.x_list)):
                        # pred, h_new, c_new = self.step(label, length, x, last, j, h, c)
                        pred = self.step_w_hidden(hidden_output, last, j)
                        sam.append(pred.multinomial(1))
                        last = torch.cat(sam,dim=1)
                    first = sam
                if i != 0:
                    for j in range(len(self.x_list) - len(META_LIST)):
                        pred = self.step_w_hidden(hidden_output, last, j)
                        sam.append(pred.multinomial(1))
                        last = torch.cat(sam,dim=1)
                    sam = sam + first[-len(META_LIST):]
                    last = torch.cat(sam,dim=1)
                x = last
                # h = h_new
                # c = c_new
                samples.append(x)
        else:
            # print(x.shape)
            given_len = x.size(1)
            lis = x.chunk(x.size(1), dim=1)
            first_tensor = lis[0].squeeze(1)
            sam = []
            for i in range(given_len):
                input = lis[i].squeeze(1)
                # for j in range(len(self.x_list)):
                #     if j == 0:
                # pred, h_new, c_new = self.step(label, length, input, None, 0, h, c)
                hidden_output, h, c = self.get_LSTM_hidden(label, length, input, h, c)
                    # else:
                    #     pred, h_new, c_new = self.step(label, length, input, input[:,:j], j, h, c)
                if i == given_len - 1:
                    pred = self.step_w_hidden(hidden_output, None, 0)
                    sam.append(pred.multinomial(1))
                    for j in range(1,len(self.x_list)- len(META_LIST)):
                        pred = self.step_w_hidden(hidden_output, input[:,:j], j)
                        # pred, h_new, c_new = self.step(label, length, input, input[:,:j], j, h, c)
                        sam.append(pred.multinomial(1))
                    # sam = sam + first[-len(META_LIST):]
                samples.append(input)
                # h = h_new
                # c = c_new
  
            sam_tensor = torch.cat(sam,dim=1)
            sam_tensor = torch.cat([sam_tensor, first_tensor[:,-len(META_LIST):]], dim=1)
            # print(sam_tensor.shape)
            
            for i in range(given_len, self.max_seq_len):
                samples.append(sam_tensor)
                sam = []
                last = None
                hidden_output, h, c = self.get_LSTM_hidden(label, length, sam_tensor, h, c)
                for j in range(len(self.x_list)- len(META_LIST)):
                    pred = self.step_w_hidden(hidden_output, last, j)
                    #pred, h_new, c_new = self.step(label, length, sam_tensor, last, j, h, c)
                    sam.append(pred.multinomial(1))
                    last = torch.cat(sam,dim=1)
                last = torch.cat([last, first_tensor[:,-len(META_LIST):]], dim=1)
                sam_tensor = last
                # h = h_new
                # c = c_new
        output = torch.stack(samples, dim=1)
        # print(output.shape)
        return output