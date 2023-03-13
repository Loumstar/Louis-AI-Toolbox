from typing import Optional

import torch
import torch.nn as nn

from .attention import DotAttention
from .feed_forward import FeedForward
from .position import PositionalEncoder
from .residual_norm import ResidualNorm


class EncoderLayer(nn.Module):
    def __init__(
        self,
        dimensions: int,
        heads: int,
        feed_forward_hidden_size: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.attention = DotAttention(heads, dimensions, dropout)

        self.norm_1 = ResidualNorm(dimensions, dropout)
        self.norm_2 = ResidualNorm(dimensions, dropout)

        self.fc = FeedForward(dimensions, feed_forward_hidden_size)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attention = self.attention(x, x, x, mask=mask)

        normalised = self.norm_1(attention, x)
        out = self.fc(normalised)

        return self.norm_2(out, normalised)


class Encoder(nn.Module):
    def __init__(
        self,
        embedder: nn.Module,
        dimensions: int,
        feed_forward_hidden_size: int,
        heads: int,
        layers: int,
        max_length: int = 200,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.embed = embedder
        self.position = PositionalEncoder(dimensions, max_length, dropout)

        encoder_layers = [
            EncoderLayer(dimensions, heads, feed_forward_hidden_size, dropout)
            for _ in range(layers)
        ]

        self.layers = nn.Sequential(*encoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.embed(x)
        encodings = self.position(embeddings)

        return self.layers(encodings)
