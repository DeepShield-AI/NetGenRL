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
        freqs = torch.exp(
            -torch.arange(half, device=device) / half * torch.log(torch.tensor(10000.))
        )
        angles = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_ch)

    def forward(self, x, cond_emb):
        cond = self.cond_proj(cond_emb)[:, :, None, None]  # [B, C, 1, 1]

        h = self.conv1(x) + cond
        h = self.act(h)
        h = self.conv2(h)
        h = self.act(h)
        return h


class SimpleConditionalUNet(nn.Module):
    """
    现在支持：
    - 可变矩阵高度 H
    - 额外条件 height (流数据包数量)
    """
    def __init__(
        self,
        input_channels=1,
        base_channels=64,
        cond_dim=32,
        max_height=2048,      # 可根据你流的最大长度调整
    ):
        super().__init__()

        self.time_embed = SinusoidalTimeEmbedding(cond_dim)
        self.label_embed = nn.Embedding(100, cond_dim)
        
        # 新增：height embedding
        self.height_embed = nn.Embedding(max_height, cond_dim)

        total_cond_dim = cond_dim * 3

        self.down1 = UNetBlock(input_channels, base_channels, total_cond_dim)
        self.down2 = UNetBlock(base_channels, base_channels * 2, total_cond_dim)
        self.bot  = UNetBlock(base_channels * 2, base_channels * 2, total_cond_dim)
        self.up2  = UNetBlock(base_channels * 4, base_channels, total_cond_dim)
        self.up1  = UNetBlock(base_channels * 2, input_channels, total_cond_dim)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t, label, height):
        """
        x: (B, C, H, W)
        height: (B,)  每个样本原始高度（未 pad 之前）
        """
        temb = self.time_embed(t)
        lemb = self.label_embed(label)
        hemb = self.height_embed(height)

        cond = torch.cat([temb, lemb, hemb], dim=-1)

        # encoder
        h1 = self.down1(x, cond)
        h2 = self.down2(self.pool(h1), cond)
        h3 = self.bot(self.pool(h2), cond)

        # decoder
        u2 = F.interpolate(h3, size=h2.shape[-2:], mode="nearest")
        u2 = self.up2(torch.cat([u2, h2], dim=1), cond)

        u1 = F.interpolate(u2, size=h1.shape[-2:], mode="nearest")
        out = self.up1(torch.cat([u1, h1], dim=1), cond)

        return out
