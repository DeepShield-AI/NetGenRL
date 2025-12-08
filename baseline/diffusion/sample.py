import torch
from model.unet import SimpleConditionalUNet
from model.diffusion import Diffusion


def generate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleConditionalUNet().to(device)
    diffusion = Diffusion(model).to(device)
    diffusion.load_state_dict(torch.load("model.pth", map_location=device))

    label = torch.tensor([3], device=device)
    height = torch.tensor([40], device=device)   # 想生成高度=40 的流
    shape = (1, 1, 40, 32)

    x = diffusion.sample(label, height, shape, device=device)

    print(x)
