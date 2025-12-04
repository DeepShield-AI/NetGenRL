# train.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from unet import SimpleConditionalUNet
from diffusion import Diffusion
import torch.optim as optim


def get_dummy_dataset(num_samples=1000, H=32, W=32):
    # 随机模拟{-1,0,1}矩阵
    x = torch.randint(-1, 2, (num_samples, 1, H, W)).float()
    y = torch.randint(0, 10, (num_samples,))
    return TensorDataset(x, y)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = get_dummy_dataset()
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimpleConditionalUNet(input_channels=1).to(device)
    diffusion = Diffusion(model).to(device)

    opt = optim.Adam(diffusion.parameters(), lr=1e-4)

    for epoch in range(10):
        for x, label in loader:
            x = x.to(device)
            label = label.to(device)

            loss = diffusion.loss(x, label)
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch}: loss={loss.item():.4f}")

    torch.save(diffusion.state_dict(), "diffusion.pth")
    print("Model saved.")


if __name__ == "__main__":
    main()
