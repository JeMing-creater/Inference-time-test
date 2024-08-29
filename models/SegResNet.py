import torch
from monai.networks.nets import SegResNet
from monai.networks.nets import UNETR

if __name__ == '__main__':
    x = torch.randn(size=(1, 4, 128, 128, 128))
    model = SegResNet(
        spatial_dims=3,
        init_filters=24,
        in_channels=4,
        out_channels=3
    )
    print(model(x).size())