# model/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half, device=device) / half * torch.log(torch.tensor(10000.)))
        angles = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.cond_proj = nn.Linear(cond_dim, out_ch)

    def forward(self, x, cond_emb):
        h = self.conv1(x)
        h = h + self.cond_proj(cond_emb)[:, :, None, None]
        h = self.act(h)
        h = self.conv2(h)
        return self.act(h)


class SimpleConditionalUNet(nn.Module):
    def __init__(self, input_channels=1, base_channels=64, cond_dim=32):
        super().__init__()

        self.time_embed = SinusoidalTimeEmbedding(cond_dim)
        self.label_embed = nn.Embedding(100, cond_dim)  # 假设最多 100 类

        self.down1 = UNetBlock(input_channels, base_channels, cond_dim * 2)
        self.down2 = UNetBlock(base_channels, base_channels * 2, cond_dim * 2)
        self.bot = UNetBlock(base_channels * 2, base_channels * 2, cond_dim * 2)
        self.up2 = UNetBlock(base_channels * 4, base_channels, cond_dim * 2)
        self.up1 = UNetBlock(base_channels * 2, input_channels, cond_dim * 2)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t, label):
        temb = self.time_embed(t)              # [B, cond_dim]
        lemb = self.label_embed(label)         # [B, cond_dim]
        cond = torch.cat([temb, lemb], dim=-1) # [B, 2*cond_dim]

        # encoder
        h1 = self.down1(x, cond)
        h2 = self.down2(self.pool(h1), cond)
        h3 = self.bot(self.pool(h2), cond)

        # decoder
        u2 = F.interpolate(h3, scale_factor=2, mode="nearest")
        u2 = self.up2(torch.cat([u2, h2], dim=1), cond)

        u1 = F.interpolate(u2, scale_factor=2, mode="nearest")
        out = self.up1(torch.cat([u1, h1], dim=1), cond)

        return out  # 预测噪声
