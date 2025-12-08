# model/diffusion.py
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class Diffusion(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # linear schedule
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1 - alpha_cumprod))

    def q_sample(self, x0, t, noise=None):
        """前向扩散: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x0)

        return (
            self.sqrt_alpha_cumprod[t][:, None, None, None] * x0 +
            self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None] * noise
        )

    def p_mean_variance(self, xt, t, label, height):
        """反向扩散: μ_t"""
        noise_pred = self.model(xt, t, label, height)

        alpha_t = self.alphas[t][:, None, None, None]
        alpha_cum_t = self.alpha_cumprod[t][:, None, None, None]
        beta_t = self.betas[t][:, None, None, None]

        # μ_theta = 1/sqrt(alpha) * (xt - (1-alpha)/sqrt(1-alphaBar) * eps_theta)
        mean = (
            1 / torch.sqrt(alpha_t) *
            (xt - beta_t / torch.sqrt(1 - alpha_cum_t) * noise_pred)
        )

        return mean, beta_t

    def sample(self, label, shape, device, height):
        """依据 label 采样新矩阵"""
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.timesteps)):
            tt = torch.full((shape[0],), t, dtype=torch.long, device=device)

            mean, var = self.p_mean_variance(x, tt, label, height)

            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean

        return x

    def loss(self, x0, label, height):
        """DDPM 训练 loss：预测噪声"""
        B = x0.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)

        xt = self.q_sample(x0, t, noise)

        noise_pred = self.model(xt, t, label, height)

        return F.mse_loss(noise_pred, noise)
