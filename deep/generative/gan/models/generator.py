import torch
import torch.nn as nn

from ....resnet import ResNetBlock


class Generator(nn.Module):
    def __init__(self, latent_dimensions: int, out_channels: int) -> None:
        super().__init__()

        self.latent_dimensions = latent_dimensions
        self.scale = nn.Parameter(
            torch.tensor([1], dtype=torch.float), requires_grad=True
        )

        self.register_parameter(name="scale", param=self.scale)

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dimensions, 256, 3, 2, 0, bias=False
            ),  # 3 x 3 x 256
            ResNetBlock(256, (256, 256), (3, 3), 1),
            nn.ConvTranspose2d(
                256, 128, 3, 2, 0, bias=False, output_padding=1
            ),  # 8 x 8 x 128
            ResNetBlock(128, (128, 128), (3, 3), 1),
            nn.ConvTranspose2d(
                128, 64, 3, 2, 1, bias=False, output_padding=1
            ),  # 16 x 16 x 64
            ResNetBlock(64, (64, 64), (3, 3), 1),
            nn.ConvTranspose2d(
                64, out_channels, 3, 2, 1, bias=False, output_padding=1
            ),  # 32 x 32 x 3
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.latent_dimensions, 1, 1)
        return self.scale * self.layers(x)
