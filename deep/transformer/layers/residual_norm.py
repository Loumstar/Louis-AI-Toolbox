from typing import Tuple

import torch
import torch.nn as nn

from ...resnet import ResNetBlock


class ResidualNorm(ResNetBlock):
    def __init__(
        self,
        in_channels: int,
        channels: Tuple[int, ...],
        kernels: Tuple[int, ...],
        stride: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(in_channels, channels, kernels, stride)

        self.norm = nn.LayerNorm(channels[-1])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(super().forward(x))

        return self.dropout(out)
