import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(
        self, features: int, hidden_size: int, dropout: float = 0.3
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
