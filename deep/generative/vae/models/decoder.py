import torch
import torch.nn as nn

from ....resnet import ResNetBlock


class VAEDecoder(nn.Module):
    def __init__(self, latent_dimensions: int, out_channels: int) -> None:
        super().__init__()

        self.fc = nn.Linear(latent_dimensions, 256)

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 0, bias=False),  # 3 x 3 x 128
            ResNetBlock(128, (128, 128), (3, 3), 1),
            nn.ConvTranspose2d(
                128, 64, 3, 2, 0, bias=False, output_padding=0
            ),  # 7 x 7 x 64
            ResNetBlock(64, (64, 64), (3, 3), 1),
            nn.ConvTranspose2d(
                64, 32, 3, 2, 1, bias=False, output_padding=1
            ),  # 14 x 14 x 32
            ResNetBlock(32, (32, 32), (3, 3), 1),
            nn.ConvTranspose2d(
                32, out_channels, 3, 2, 1, bias=False, output_padding=1
            ),  # 28 x 28 x 1
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.fc(x).view(x.size(0), -1, 1, 1)
        return self.layers(features)
