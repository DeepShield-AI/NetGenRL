import torch
from torch.utils.data import DataLoader
from model.unet import SimpleConditionalUNet
from model.diffusion import Diffusion
import torch.optim as optim
import torch.nn.functional as F


def pad_batch(batch_x):
    """
    batch_x: list of Tensors with shapes (1, H, W)
    返回 pad 后的 (B, C, H_max, W), 和 heights 向量
    """
    heights = torch.tensor([x.shape[1] for x in batch_x], dtype=torch.long)
    max_h = max(heights).item()

    padded = []
    for x in batch_x:
        H = x.shape[1]
        pad_h = max_h - H
        # pad = (W_left, W_right, H_top, H_bottom)
        x_pad = F.pad(x, (0, 0, 0, pad_h))
        padded.append(x_pad)

    return torch.stack(padded, dim=0), heights


def dummy_dataset(num=1000):
    # 随机生成不同高度的矩阵
    W = 32
    data = []
    for _ in range(num):
        H = torch.randint(16, 64, (1,)).item()
        x = torch.randint(-1, 2, (1, H, W)).float()
        label = torch.randint(0, 10, (1,)).item()
        data.append((x, label, H))
    return data


def collate_fn(batch):
    xs = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    heights = torch.tensor([b[2] for b in batch], dtype=torch.long)
    x_padded, heights = pad_batch(xs)
    return x_padded, labels, heights


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = dummy_dataset()
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    model = SimpleConditionalUNet(input_channels=1).to(device)
    diffusion = Diffusion(model).to(device)
    opt = optim.Adam(diffusion.parameters(), lr=1e-4)

    for epoch in range(5):
        for x, label, height in loader:
            x = x.to(device)
            label = label.to(device)
            height = height.to(device)

            loss = diffusion.loss(x, label, height)
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch}, loss={loss.item():.4f}")

    torch.save(diffusion.state_dict(), "model.pth")

