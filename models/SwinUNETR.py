import torch
from monai.networks.nets import SwinUNETR


if __name__ == '__main__':
    x = torch.randn(size=(1, 4, 128, 128, 128))
    model = SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=3,
        feature_size=48
    )
    print(model(x).size())