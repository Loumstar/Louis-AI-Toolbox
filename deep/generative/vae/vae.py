import torch
import torch.nn as nn

from .models.decoder import VAEDecoder
from .models.encoder import VAEEncoder


class VariationalAutoEncoder(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, latent_dimensions: int
    ) -> None:
        super().__init__()

        self.encoder = VAEEncoder(in_channels, latent_dimensions)
        self.decoder = VAEDecoder(latent_dimensions, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)

        return reconstruction
