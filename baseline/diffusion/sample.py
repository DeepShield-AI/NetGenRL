# sample.py
import torch
from model.unet import SimpleConditionalUNet
from model.diffusion import Diffusion


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleConditionalUNet(input_channels=1).to(device)
    diffusion = Diffusion(model).to(device)
    diffusion.load_state_dict(torch.load("diffusion.pth", map_location=device))

    label = torch.tensor([3], device=device)  # 生成 label=3 的矩阵
    x = diffusion.sample(label, shape=(1, 1, 32, 32), device=device)

    print("Generated matrix:", x[0, 0])


if __name__ == "__main__":
    main()
