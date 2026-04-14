import torch
import torch.nn as nn
import torch.nn.functional as F


class TransGenerator(nn.Module):
    def __init__(self, label_dim, seq_dim, max_seq_len, x_list, device):
        super().__init__()

        self.label_dim = label_dim
        self.seq_dim = seq_dim
        self.max_seq_len = max_seq_len
        self.x_list = x_list
        self.pred_dim = sum(x_list)
        self.device = device

        # ===== 维度 =====
        self.hidden_dim = 512
        self.embedding_dim = 128
        self.attribute_dim = 64
        self.state_dim = 64
        self.noise_dim = 32
        self.length_emb_dim = 64
        self.noise_pos_emb_dim = 16
        self.transformer_layers = 4

        # ===== 共享 Embedding =====
        self.emb = nn.ModuleList([
            nn.Embedding(x_len, self.embedding_dim) for x_len in x_list
        ])

        self.pad_embedding = nn.Parameter(torch.zeros(1, self.attribute_dim))

        # ===== 字段内自回归 =====
        self.condition_fix = nn.ModuleList([
            nn.Sequential(
                nn.Linear((i + 1) * self.embedding_dim, self.attribute_dim),
                nn.ReLU()
            ) for i in range(len(x_list) - 1)
        ])

        # ===== length =====
        self.length_fc = nn.Sequential(
            nn.Linear(max_seq_len, self.length_emb_dim),
            nn.ReLU()
        )

        # =====================================================
        # ✅ label-controlled heads（核心）
        # =====================================================

        # 输入头（类似原 combine_fc）
        self.input_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_dim * len(x_list) + self.length_emb_dim, self.hidden_dim),
                nn.ReLU()
            ) for _ in range(label_dim)
        ])

        # ===== Transformer =====
        self.pos_emb = nn.Embedding(max_seq_len, self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,
            batch_first=True,  # allow (B, S, E)
            activation="relu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_layers)

        # ===== 状态头（IB）=====
        self.state_mu = nn.Linear(self.hidden_dim, self.state_dim)
        self.state_logvar = nn.Linear(self.hidden_dim, self.state_dim)
  
        # ===== 噪声头（关键：不依赖历史）=====
        # self.pos_fc = nn.Sequential(
        #     nn.Linear(max_seq_len, self.noise_pos_emb_dim),
        #     nn.ReLU()
        # )
        
        # self.noise_heads = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(self.noise_dim + self.length_emb_dim + self.noise_pos_emb_dim, self.noise_dim),  # eps + pos + length
        #         nn.ReLU()
        #     ) for _ in range(label_dim)
        # ])
        self.noise_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.noise_dim + self.length_emb_dim, self.noise_dim),  # eps + pos + length
                nn.ReLU()
            ) for _ in range(label_dim)
        ])

        # ===== 输出层 =====
        self.output_layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.state_dim + self.noise_dim + self.attribute_dim, x_len),
                nn.Identity()
            ) for x_len in x_list
        ])
        
        self.softmax = nn.LogSoftmax(dim=-1)

    # =====================================================
    # forward
    # =====================================================
    def forward(self, label, length, x, x_now):

        B, S, _ = x.shape
        label_int = torch.argmax(label.clone(), dim=1)

        # ===== length =====
        length_one_hot = F.one_hot((length.clone().detach().long() - 1), num_classes=self.max_seq_len).float().to(self.device)
        length_out = self.length_fc(length_one_hot)
        length_expand = length_out.unsqueeze(1).expand(-1, S, -1)

        # ===== embedding =====
        emb_list = torch.cat([self.emb[i](x[:, :, i]) for i in range(len(self.x_list))], dim=2)

        combined = torch.cat([emb_list, length_expand], dim=2)

        # ===== label 路由 =====
        head_outputs = torch.stack([self.input_heads[idx](combined) for idx in range(self.label_dim)],dim=1)  # (B, L, S, H)

        index = label_int.view(B, 1, 1, 1).expand(-1, 1, S, self.hidden_dim)
        combined_out = torch.gather(head_outputs, 1, index).squeeze(1)
        seq_len = combined_out.size(1)
        # ===== Transformer =====
        pos_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(B, -1)
        combined_out = combined_out + self.pos_emb(pos_ids)

        mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
        hidden = self.transformer(combined_out, mask=mask)

        # =====================================================
        # ✅ 状态（IB）
        # =====================================================
        # mu_all = torch.stack([head(hidden) for head in self.state_mu_heads], dim=1)
        # logvar_all = torch.stack([head(hidden) for head in self.state_logvar_heads], dim=1)

        # index_state = label_int.view(B, 1, 1, 1).expand(-1, 1, S, self.state_dim)

        # mu = torch.gather(mu_all, 1, index_state).squeeze(1)
        # logvar = torch.gather(logvar_all, 1, index_state).squeeze(1)
        mu = self.state_mu(hidden)          # (B,S,state_dim)
        logvar = self.state_logvar(hidden)  # (B,S,state_dim)
        std = torch.exp(0.5 * logvar)
        eps_state = torch.randn_like(std)
        s_t = mu + eps_state * std

        # =====================================================
        # ✅ 噪声（只依赖 t, label, length）
        # =====================================================
        eps = torch.randn(B, S, self.noise_dim).to(self.device)

        # pos_norm = (pos_ids.float() / self.max_seq_len).unsqueeze(-1)
        # length_norm = (length.unsqueeze(1).expand(-1, S) / self.max_seq_len).unsqueeze(-1)
        # pos_emb_noise = self.pos_fc(pos_ids.float())  # (B, S, noise_pos_emb_dim)
        # pos_emb_noise_expand = pos_emb_noise.unsqueeze(1).expand(-1, S, -1)
        # noise_input = torch.cat([eps, length_expand, pos_emb_noise_expand], dim=2)
        
        noise_input = torch.cat([eps, length_expand], dim=2)

        noise_all = torch.stack([self.noise_heads[i](noise_input) for i in range(self.label_dim)],dim=1)
        index_noise = label_int.view(B, 1, 1, 1).expand(-1, 1, S, self.noise_dim)
        z_t = torch.gather(noise_all, 1, index_noise).squeeze(1)

        # ===== attribute =====
        now_emb = torch.cat([self.emb[i](x_now[:, :, i]) for i in range(len(self.x_list) - 1)], dim=2)

        # attr = [self.pad_embedding.expand(B, S, -1)]
        # for i in range(len(self.x_list) - 1):
        #     attr.append(
        #         self.condition_fix[i](
        #             now_emb[:, :, :(i + 1) * self.embedding_dim]
        #         )
        #     )
        
        attribute_features = [self.condition_fix[i](now_emb[:,:,:(i+1)*self.embedding_dim]) for i in range(len(self.x_list) - 1)]
        attribute_features = [self.pad_embedding.expand(x_now.shape[0],x_now.shape[1],self.attribute_dim)] + attribute_features

        # ===== decode =====
        preds = []
        for i in range(len(self.x_list)):
            h = torch.cat([s_t, z_t, attribute_features[i]], dim=2)
            out = self.output_layer[i](h.reshape(-1, h.shape[-1]))
            preds.append(F.log_softmax(out, dim=-1))

        preds = torch.cat(preds, dim=1).view(B, S, self.pred_dim)

        return {
            "preds": preds,
            "mu": mu,
            "logvar": logvar,
            "state": s_t,
            "noise": z_t
        }

    # =====================================================
    # get_hidden（用于生成）
    # =====================================================
    def get_hidden(self, label, length, x):

        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            x_seq = x                # (B,S,seq_dim)
        else:
            raise RuntimeError("Unsupported x shape in get_hidden: %s" % (x.shape,))

        B, S, _ = x.shape
        label_int = torch.argmax(label.clone(), dim=1)

        length_one_hot = F.one_hot((length.clone().detach().long() - 1), num_classes=self.max_seq_len).float().to(self.device)
        length_out = self.length_fc(length_one_hot)
        length_expand = length_out.unsqueeze(1).expand(-1, S, -1)

        emb_list = torch.cat([self.emb[i](x[:, :, i]) for i in range(len(self.x_list))], dim=2)
        combined = torch.cat([emb_list, length_expand], dim=2)

        head_outputs = torch.stack([self.input_heads[i](combined) for i in range(self.label_dim)],dim=1)

        index = label_int.view(B, 1, 1, 1).expand(-1, 1, S, self.hidden_dim)
        combined_out = torch.gather(head_outputs, 1, index).squeeze(1)

        seq_len = combined_out.size(1)
        pos_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(B, -1)
        combined_out = combined_out + self.pos_emb(pos_ids)

        mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
        hidden = self.transformer(combined_out, mask=mask)

        # mu_all = torch.stack([head(hidden) for head in self.state_mu_heads], dim=1)

        # index_state = label_int.view(B, 1, 1, 1).expand(-1, 1, S, self.state_dim)
        # mu = torch.gather(mu_all, 1, index_state).squeeze(1)
        mu = self.state_mu(hidden)          # (B,S,state_dim)
        # logvar = self.state_logvar(hidden)  # (B,S,state_dim)

        return mu[:, -1, :]  # 用均值
    
    def get_noise(self, label, length, seq_len):
        B = label.size(0)
        label_int = torch.argmax(label.clone(), dim=1)

        length_one_hot = F.one_hot((length.clone().detach().long() - 1), num_classes=self.max_seq_len).float().to(self.device)
        length_out = self.length_fc(length_one_hot)
        length_expand = length_out.unsqueeze(1).expand(-1, seq_len, -1)

        # pos_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(B, -1)
        # print(pos_ids.shape)
        # pos_emb_noise = self.pos_fc(pos_ids.float())  # (B, S, noise_pos_emb_dim)
        # pos_emb_noise_expand = pos_emb_noise.unsqueeze(1).expand(-1, seq_len, -1)

        eps = torch.randn(B, seq_len, self.noise_dim).to(self.device)
        # noise_input = torch.cat([eps, length_expand, pos_emb_noise_expand], dim=2)
        noise_input = torch.cat([eps, length_expand], dim=2)

        noise_all = torch.stack([self.noise_heads[i](noise_input) for i in range(self.label_dim)],dim=1)
        index_noise = label_int.view(B, 1, 1, 1).expand(-1, 1, seq_len, self.noise_dim)
        z_t = torch.gather(noise_all, 1, index_noise).squeeze(1)

        return z_t[:, -1, :]  # 返回最后一个时间步的噪声

    # =====================================================
    # step（字段生成）
    # =====================================================
    def step_w_hidden(self, s_t, z_t, x_now, dim_now):

        if dim_now > 0 and x_now is not None:
            now_emb = torch.cat([self.emb[i](x_now[:, i]) for i in range(dim_now)], dim=1)
            attr = self.condition_fix[dim_now - 1](now_emb)
        else:
            attr = self.pad_embedding.expand(s_t.shape[0], -1)

        h = torch.cat([s_t, z_t, attr], dim=1)
        out = self.output_layer[dim_now](h)

        return F.softmax(out, dim=-1)

    # =====================================================
    # sample
    # =====================================================
    def sample(self, batch_size, label, length, x=None):
        flag = False # whether sample from zero
        if x is None:
            flag = True
        if flag:
            x = torch.zeros((batch_size,1,self.seq_dim)).long().to(self.device)
            
        samples = []
        if flag:
            for i in range(self.max_seq_len): 
                last = None
                step_out = []
                st = self.get_hidden(label, length, x)
                zt = self.get_noise(label, length, i+1)
                for j in range(len(self.x_list)):
                    pred = self.step_w_hidden(st, zt, last, j)
                    step_out.append(pred.multinomial(1))
                    last = torch.cat(step_out, dim=1)
                x = torch.cat([x,last.unsqueeze(1)],dim = 1)
                # samples.append(x)
                samples.append(last)
        else:
            given_len = x.size(1)
            lis = x.chunk(x.size(1), dim=1)
            # sam = []
            for i in range(given_len):
                input = lis[i].squeeze(1)
                samples.append(input)
            sam_tensor = x
            
            for i in range(given_len, self.max_seq_len):
                # samples.append(sam_tensor)
                step_out = []
                last = None
                st = self.get_hidden(label, length, sam_tensor)
                zt = self.get_noise(label, length, i+1)
                for j in range(len(self.x_list)):
                    pred = self.step_w_hidden(st, zt, last, j)
                    step_out.append(pred.multinomial(1))
                    last = torch.cat(step_out, dim=1)
                # sam_tensor = last
                sam_tensor = torch.cat([sam_tensor,last.unsqueeze(1)],dim = 1)
                samples.append(last)
        output = torch.stack(samples, dim=1)
        return output