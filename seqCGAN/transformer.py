import torch
import torch.nn as nn
import torch.nn.functional as F

class TransGenerator(nn.Module):
    def __init__(self, label_dim, seq_dim, max_seq_len, x_list, device):
        super(TransGenerator, self).__init__()
        
        self.label_dim = label_dim
        self.hidden_dim = 512
        # self.lstm_layers = 4
        self.transformer_layers = 4
        self.embedding_dim = 128
        self.attribute_dim = 64
        self.max_seq_len = max_seq_len
        self.seq_dim = seq_dim
        self.pred_dim = sum(x_list)
        self.x_list = x_list # list of seq value dim
        self.device = device
        self.count_list = []
                
        self.emb = nn.ModuleList([
            nn.Embedding(x_len, self.embedding_dim) for x_len in x_list
        ]) 
        
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
        
        self.combine_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128*len(x_list)+64,self.hidden_dim),
                nn.ReLU(True)
            ) for _ in range(self.label_dim)
        ])
        
        self.pos_emb = nn.Embedding(self.max_seq_len, self.hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,
            batch_first=True,  # allow (B, S, E)
            activation="relu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_layers)
        
        # self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.lstm_layers, batch_first=True)
        # self.lstm.flatten_parameters()

        self.output_layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim + self.attribute_dim, x_len),
                nn.Identity()
            ) for x_len in x_list
        ])
        
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def init_hidden(self, batch_size):
        # h = torch.zeros((self.lstm_layers, batch_size, self.hidden_dim),device=self.device)
        # c = torch.zeros((self.lstm_layers, batch_size, self.hidden_dim),device=self.device)
        # return h, c
        return None,None
        
    def forward(self, label, length, x, x_now): # x: (batch_size, seq_len, seq_dim)
        # self.lstm.flatten_parameters()
        length_one_hot = F.one_hot((length.clone().detach().long() - 1), num_classes=self.max_seq_len).float().to(self.device)

        label_int = torch.argmax(label.clone(),1)
        length_out = self.length_fc(length_one_hot)

        emb_list = torch.cat([self.emb[i](x[:,:,i]) for i in range(len(self.x_list))], dim=2)
        
        now_emb_list = torch.cat([self.emb[i](x_now[:,:,i]) for i in range(len(self.x_list)-1)], dim=2)
        attribute_features = [self.condition_fix[i](now_emb_list[:,:,:(i+1)*self.embedding_dim]) for i in range(len(self.x_list) - 1)]
        attribute_features = [self.pad_embedding.expand(x_now.shape[0],x_now.shape[1],self.attribute_dim)] + attribute_features
        
        length_expand = length_out.unsqueeze(1).expand(-1, emb_list.size(1), -1)

        combined = torch.cat([emb_list, length_expand], dim=2)
        
        fc_outputs = torch.stack([self.combine_fc[idx](combined) for idx in range(len(self.combine_fc))], dim=1)
        
        indices = label_int.view(-1, 1, 1, 1).expand(-1, -1, fc_outputs.size(2), fc_outputs.size(3))

        combined_out = torch.gather(fc_outputs, dim=1, index=indices).squeeze(1)
        
        seq_len = combined_out.size(1)
        pos_ids = torch.arange(seq_len, device=combined_out.device).unsqueeze(0).expand(combined_out.size(0), -1)
        pos_embeddings = self.pos_emb(pos_ids)  # (B, S, hidden_dim)
        combined_out = combined_out + pos_embeddings

        # h0, c0 = self.init_hidden(combined_out.size(0))
        # hidden_output, (h, c) = self.lstm(combined_out, (h0, c0))
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        hidden_output = self.transformer(combined_out,mask = mask)

        preds = []
        for i in range(len(self.x_list)):
            hidden_w_attr = torch.cat([hidden_output, attribute_features[i]], dim=2)
            output = self.output_layer[i](hidden_w_attr.contiguous().view(-1, self.hidden_dim + self.attribute_dim))
            pred = self.softmax(output)
            preds.append(pred)
            
        preds = torch.cat(preds, dim=1).view(-1, self.max_seq_len, self.pred_dim) 
        return preds
    
    def get_LSTM_hidden(self, label, length, x, h, c):
        
        length_one_hot = F.one_hot((length.clone().detach().long() - 1), num_classes=self.max_seq_len).float().to(self.device)
        
        length_out = self.length_fc(length_one_hot)
        label_int = torch.argmax(label.clone(),1)
        
        if x.dim() == 2:
            x_seq = x.unsqueeze(1)   # (B,1,seq_dim)
        elif x.dim() == 3:
            x_seq = x                # (B,S,seq_dim)
        else:
            raise RuntimeError("Unsupported x shape in get_LSTM_hidden: %s" % (x.shape,))
        
        B, S, T = x_seq.shape
        
        # print(B,S,T)

        # build per-timestep embeddings: (B, S, embedding_dim * len(x_list))
        # emb_per_field = [self.emb[i](x_seq[:, :, i]) for i in range(len(self.x_list)) ]  # each (B,S,emb)
        # emb_list = torch.cat(emb_per_field, dim=2)   # (B, S, emb_total)
        
        # print(emb_list.shape)
        
        # # emb_list = torch.cat([self.emb[i](x[:,i]) for i in range(len(self.x_list))], dim=1)
        # length_expand = length_out.unsqueeze(1).expand(-1, S, -1)   # (B, S, 64)
        
        # print(length_expand.shape)
        # # combined = torch.cat([emb_list, length_out], dim=1)
        # combined = torch.cat([emb_list, length_expand], dim=2)
        # print(combined.shape)

        # fc_outputs = torch.stack([self.combine_fc[idx](combined) for idx in range(len(self.combine_fc))], dim=2)
        
        # print(fc_outputs.shape)
        
        # indices = label_int.view(-1, 1, 1).expand(-1, -1, fc_outputs.size(2), fc_outputs.size(3))

        # combined_out = torch.gather(fc_outputs, dim=2, index=indices)
        
        # pos_ids = torch.arange(S, device=self.device).unsqueeze(0).expand(B, -1)  # (B, S)
        # pos_embeddings = self.pos_emb(pos_ids)  # (B, S, hidden_dim)
        # combined_out = combined_out + pos_embeddings
        # hidden_seq = self.transformer(combined_out)  # (B,1,hidden_dim)
        
        emb_list = torch.cat([self.emb[i](x_seq[:,:,i]) for i in range(len(self.x_list))], dim=2)
        
        length_expand = length_out.unsqueeze(1).expand(-1, emb_list.size(1), -1)

        combined = torch.cat([emb_list, length_expand], dim=2)
        
        fc_outputs = torch.stack([self.combine_fc[idx](combined) for idx in range(len(self.combine_fc))], dim=1)
        
        indices = label_int.view(-1, 1, 1, 1).expand(-1, -1, fc_outputs.size(2), fc_outputs.size(3))
        
        combined_out = torch.gather(fc_outputs, dim=1, index=indices).squeeze(1)
        
        seq_len = combined_out.size(1)
        pos_ids = torch.arange(seq_len, device=combined_out.device).unsqueeze(0).expand(combined_out.size(0), -1)
        pos_embeddings = self.pos_emb(pos_ids)  # (B, S, hidden_dim)
        combined_out = combined_out + pos_embeddings

        # h0, c0 = self.init_hidden(combined_out.size(0))
        # hidden_output, (h, c) = self.lstm(combined_out, (h0, c0))
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        hidden_seq = self.transformer(combined_out,mask=mask)
        hidden_output = hidden_seq[:,-1,:]  # (B, hidden_dim)
        
        # combined_out_step = combined_out.unsqueeze(1)  # (B,1,hidden_dim)
        # pos_embeddings = self.pos_emb(torch.zeros((combined_out_step.size(0), 1), dtype=torch.long, device=combined_out_step.device))
        # combined_out_step = combined_out_step + pos_embeddings
        # hidden_seq = self.transformer(combined_out_step)  # (B,1,hidden_dim)
        # hidden_output = hidden_seq.squeeze(1)  # (B, hidden_dim)

        # hidden_output, (h, c) = self.lstm(combined_out, (h, c))
        
        # hidden_output = hidden_output.squeeze(1)        
            
        return hidden_output, None, None
    
    def step_w_hidden(self, hidden_output, x_now, dim_now):
        if dim_now > 0 and x_now is not None:
            now_emb_list = torch.cat([self.emb[i](x_now[:,i]) for i in range(dim_now)], dim=1)
            attribute_feature = self.condition_fix[dim_now - 1](now_emb_list)
        else:
            attribute_feature = self.pad_embedding.expand(hidden_output.shape[0],self.attribute_dim)
            
        hidden_w_attr = torch.cat([hidden_output, attribute_feature], dim=1)
        output = self.output_layer[dim_now](hidden_w_attr.contiguous())
        pred = F.softmax(output,dim=-1)
        return pred
    
    def sample(self, batch_size, label, length, x=None):
        flag = False # whether sample from zero
        if x is None:
            flag = True
        if flag:
            x = torch.zeros((batch_size,1,self.seq_dim)).long().to(self.device)
            
        h, c = self.init_hidden(batch_size)
        samples = []
        if flag:
            for i in range(self.max_seq_len): 
                last = None
                sam = []
                hidden_output, h, c = self.get_LSTM_hidden(label, length, x, h, c)
                for j in range(len(self.x_list)):
                    pred = self.step_w_hidden(hidden_output, last, j)
                    sam.append(pred.multinomial(1))
                    last = torch.cat(sam,dim=1)
                x = torch.cat([x,last.unsqueeze(1)],dim = 1)
                # samples.append(x)
                samples.append(last)
        else:
            given_len = x.size(1)
            lis = x.chunk(x.size(1), dim=1)
            # sam = []
            for i in range(given_len):
                input = lis[i].squeeze(1)
                # hidden_output, h, c = self.get_LSTM_hidden(label, length, input, h, c)
                # if i == given_len - 1:
                #     pred = self.step_w_hidden(hidden_output, None, 0)
                #     sam.append(pred.multinomial(1))
                #     for j in range(1,len(self.x_list)):
                #         pred = self.step_w_hidden(hidden_output, input[:,:j], j)
                #         sam.append(pred.multinomial(1))
                samples.append(input)
            # hidden_output, h, c = self.get_LSTM_hidden(label, length, x, h, c)
  
            # sam_tensor = torch.cat(sam,dim=1)
            sam_tensor = x
            
            for i in range(given_len, self.max_seq_len):
                # samples.append(sam_tensor)
                sam = []
                last = None
                hidden_output, h, c = self.get_LSTM_hidden(label, length, sam_tensor, h, c)
                for j in range(len(self.x_list)):
                    pred = self.step_w_hidden(hidden_output, last, j)
                    sam.append(pred.multinomial(1))
                    last = torch.cat(sam,dim=1)
                # sam_tensor = last
                sam_tensor = torch.cat([sam_tensor,last.unsqueeze(1)],dim = 1)
                samples.append(last)
        output = torch.stack(samples, dim=1)
        return output
    
    # def step(self, label, length, x, x_now, dim_now, h, c):
    #     """
    #     Args:
    #         x: (batch_size, seq_dim), sequence of tokens generated by generator
    #         h: (1, batch_size, hidden_dim), lstm hidden state
    #         c: (1, batch_size, hidden_dim), lstm cell state
    #     """
    #     length_one_hot = F.one_hot((length.clone().detach().long() - 1), num_classes=self.max_seq_len).float().to(self.device)
        
    #     length_out = self.length_fc(length_one_hot)
    #     label_int = torch.argmax(label.clone(),1)
        
    #     emb_list = torch.cat([self.emb[i](x[:,i]) for i in range(len(self.x_list))], dim=1)
        
    #     if dim_now > 0 and x_now is not None:
    #         now_emb_list = torch.cat([self.emb[i](x_now[:,i]) for i in range(dim_now)], dim=1)
    #         attribute_feature = self.condition_fix[dim_now - 1](now_emb_list)
    #     else:
    #         attribute_feature = self.pad_embedding.expand(x.shape[0],self.attribute_dim)
 
    #     combined = torch.cat([emb_list, length_out], dim=1)

    #     fc_outputs = torch.stack([self.combine_fc[idx](combined) for idx in range(len(self.combine_fc))], dim=1)
        
    #     indices = label_int.view(-1, 1, 1).expand(-1, -1, fc_outputs.size(2))

    #     combined_out = torch.gather(fc_outputs, dim=1, index=indices)

    #     # hidden_output, (h, c) = self.lstm(combined_out, (h, c))
        
    #     # hidden_output = hidden_output.squeeze(1)
        
    #     combined_out_step = combined_out.unsqueeze(1)  # (B,1,hidden_dim)
    #     pos_embeddings = self.pos_emb(torch.zeros((combined_out_step.size(0), 1), dtype=torch.long, device=combined_out_step.device))
    #     combined_out_step = combined_out_step + pos_embeddings
    #     hidden_seq = self.transformer(combined_out_step)  # (B,1,hidden_dim)
    #     hidden_output = hidden_seq.squeeze(1)  # (B, hidden_dim)
        
    #     output = self.output_layer[dim_now](hidden_output.view(-1, self.hidden_dim))
    #     pred = F.softmax(output,dim=-1)
                    
    #     return pred, None, None
    
    
    # def sample(self, batch_size, label, length, x=None):
    #     flag = False # whether sample from zero
    #     if x is None:
    #         flag = True
    #     if flag:
    #         # x = torch.zeros((batch_size, self.seq_dim)).long().to(self.device)
    #         history = torch.zeros((batch_size, 1, self.seq_dim), dtype=torch.long, device=self.device)
    #         given_len = 0
    #     else:
    #         if x.dim() == 3:
    #             history = x.to(self.device)
    #             given_len = history.size(1)
    #         else:
    #             given_len = x.size(1)
    #             lis = x.chunk(x.size(1), dim=1)
    #             seqs = [li.squeeze(1) for li in lis]  # list of (B, seq_dim)
    #             history = torch.stack(seqs, dim=1).to(self.device)  # (B, given_len, seq_dim)
            
    #     # h, c = self.init_hidden(batch_size)
    #     h, c = None, None
    #     samples = []
        
    #     if given_len > 0:
    #         for t in range(given_len):
    #             samples.append(history[:, t, :])

    #     # now continue generation until we have max_seq_len tokens
    #     cur_len = history.size(1)
    #     while cur_len < self.max_seq_len:
    #         # get last hidden by feeding the whole history sequence into transformer
    #         hidden_output, _, _ = self.get_LSTM_hidden(label, length, history, None, None)  # (B, hidden_dim)

    #         # autoregressively produce a full token consisting of len(x_list) fields
    #         last = None
    #         sam_fields = []
    #         for j in range(len(self.x_list)):
    #             # x_now: for j>0 we pass previous fields inside same token; shaped (B, j)
    #             x_now = last if (last is not None) else None
    #             pred = self.step_w_hidden(hidden_output, x_now, j)  # (B, vocab_j)
    #             sel = pred.multinomial(1)  # (B,1)
    #             sel = sel.long()
    #             sam_fields.append(sel.squeeze(1))  # keep as (B,)
    #             # update last: concatenate sampled fields so far for current token
    #             if last is None:
    #                 last = sel.squeeze(1)
    #             else:
    #                 last = torch.cat([last, sel.squeeze(1)], dim=1)  # (B, j+1)
    #         history = torch.cat([history, last.unsqueeze(1)], dim=1)  # (B, cur_len+1, seq_dim)
    #         samples.append(last)
    #         cur_len += 1

    #     # stack samples into (B, max_seq_len, seq_dim)
    #     output = torch.stack(samples, dim=1)
    #     return output
    #     # if given_len > 0:
    #     #     for i in range(self.max_seq_len): 
    #     #         last = None
    #     #         sam = []
    #     #         hidden_output, h, c = self.get_LSTM_hidden(label, length, x, h, c)
    #     #         for j in range(len(self.x_list)):
    #     #             pred = self.step_w_hidden(hidden_output, last, j)
    #     #             sam.append(pred.multinomial(1))
    #     #             last = torch.cat(sam,dim=1)
    #     #         x = last
    #     #         samples.append(x)
    #     # else:
    #     #     given_len = x.size(1)
    #     #     lis = x.chunk(x.size(1), dim=1)
    #     #     sam = []
    #     #     for i in range(given_len):
    #     #         input = lis[i].squeeze(1)
    #     #         hidden_output, h, c = self.get_LSTM_hidden(label, length, input, h, c)
    #     #         if i == given_len - 1:
    #     #             pred = self.step_w_hidden(hidden_output, None, 0)
    #     #             sam.append(pred.multinomial(1))
    #     #             for j in range(1,len(self.x_list)):
    #     #                 pred = self.step_w_hidden(hidden_output, input[:,:j], j)
    #     #                 sam.append(pred.multinomial(1))
    #     #         samples.append(input)
  
    #     #     sam_tensor = torch.cat(sam,dim=1)
            
    #     #     for i in range(given_len, self.max_seq_len):
    #     #         samples.append(sam_tensor)
    #     #         sam = []
    #     #         last = None
    #     #         hidden_output, h, c = self.get_LSTM_hidden(label, length, sam_tensor, h, c)
    #     #         for j in range(len(self.x_list)):
    #     #             pred = self.step_w_hidden(hidden_output, last, j)
    #     #             sam.append(pred.multinomial(1))
    #     #             last = torch.cat(sam,dim=1)
    #     #         sam_tensor = last
    #     # output = torch.stack(samples, dim=1)
    #     # return output