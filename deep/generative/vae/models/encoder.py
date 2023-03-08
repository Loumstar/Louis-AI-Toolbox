import torch
import torch.nn as nn

from ....resnet import ResNetBlock


class VAEEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dimensions: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            ResNetBlock(in_channels, (64, 64), (3, 3), 2),  # 14 x 14 x 64
            ResNetBlock(64, (128, 128), (3, 3), 2),  # 7 x 7 x 256
            ResNetBlock(128, (256, 256), (3, 3), 2),  # 3 x 3 x 512
            nn.AvgPool2d(kernel_size=3),  # 1 x 1 x 1024
        )

        self.mu = nn.Linear(256, latent_dimensions)  # mean
        self.sigma = nn.Linear(256, latent_dimensions)  # log(var)

        self.__mean = None
        self.__logvar = None

    @property
    def mean(self) -> torch.Tensor:
        if self.__mean is None:
            raise RuntimeError("Mean has not been calculated.")

        return self.__mean

    @property
    def logvar(self) -> torch.Tensor:
        if self.__logvar is None:
            raise RuntimeError("Log variance has not been calculated.")

        return self.__logvar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x).view(x.size(0), -1)
        mean, logvar = self.mu(out), self.sigma(out)

        embedding = (
            mean + torch.rand_like(mean).mul(torch.exp(logvar / 2))
            if self.training
            else mean
        )

        self.__mean = mean
        self.__logvar = logvar

        return embedding
