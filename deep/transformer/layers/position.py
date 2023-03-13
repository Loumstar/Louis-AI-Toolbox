import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(
        self, dimensions: int, max_length: int, dropout: float = 0.3
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.encodings = torch.zeros((1, max_length, dimensions))

        position = torch.arange(0, max_length).unsqueeze(1)
        steps = torch.arange(0, dimensions, step=2)

        divisor = torch.pow(1000, steps / torch.tensor([dimensions]))

        self.encodings[:, :, 0::2] = torch.sin(position / divisor)
        self.encodings[:, :, 1::2] = torch.cos(position / divisor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoding = self.encodings[:, : x.size(1)]
        return self.dropout(x + encoding.detach())
