import torch
import torch.nn as nn

from ....resnet import ResNetBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            ResNetBlock(in_channels, (64,), (3,), 2),  # 16 x 16 x 64
            ResNetBlock(64, (128,), (3,), 2),  # 8 x 8 x 128
            ResNetBlock(128, (256,), (3,), 2),  # 4 x 4 x 256
            ResNetBlock(256, (512,), (3,), 2),  # 2 x 2 x 512,
            nn.AvgPool2d(kernel_size=2),  # 1 x 1 x 512
        )

        self.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.layers(x).view(x.size(0), -1)

        return self.fc(features).view(-1)
