from typing import Optional

import torch
import torch.nn as nn

from .attention import DotAttention
from .feed_forward import FeedForward
from .position import PositionalEncoder
from .residual_norm import ResidualNorm


class DecoderLayer(nn.Module):
    def __init__(
        self,
        dimensions: int,
        heads: int,
        feed_forward_hidden_size: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.norm_1 = ResidualNorm(dimensions, dropout)
        self.norm_2 = ResidualNorm(dimensions, dropout)
        self.norm_3 = ResidualNorm(dimensions, dropout)

        self.attention_1 = DotAttention(dimensions, heads, dropout)
        self.attention_2 = DotAttention(dimensions, heads, dropout)

        self.fc = FeedForward(dimensions, feed_forward_hidden_size, dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        masked_attention = self.attention_1(x, x, x, target_mask)
        ma_normalised = self.norm_1(masked_attention, x)

        encoder_attention = self.attention_2(
            masked_attention, encoder_out, encoder_out, source_mask
        )

        ea_normalised = self.norm_2(encoder_attention, ma_normalised)

        out = self.fc(ea_normalised)

        return self.norm_3(out, ea_normalised)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        embedder: nn.Module,
        dimensions: int,
        heads: int,
        layers: int,
        feed_forward_hidden_size: int,
        max_length: int = 200,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.embed = embedder
        self.position = PositionalEncoder(dimensions, max_length, dropout)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    dimensions, heads, feed_forward_hidden_size, dropout
                )
                for _ in range(layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embedding = self.embed(x)
        encoding = self.position(embedding)

        for layer in self.layers:
            encoding = layer(encoding, encoder_out, source_mask, target_mask)

        return encoding
