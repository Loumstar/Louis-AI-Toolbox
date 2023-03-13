import torch
import torch.nn as nn


class ResidualNorm(nn.Module):
    def __init__(
        self,
        dimensions: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(dimensions)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        out = self.norm(x + residual)

        return self.dropout(out)
