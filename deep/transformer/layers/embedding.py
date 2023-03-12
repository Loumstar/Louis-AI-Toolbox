import math

import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(
        self, vocab_size: int, dimensions: int, padding_idx: int
    ) -> None:
        super().__init__()

        self.dimensions = dimensions
        self.embed = nn.Embedding(vocab_size, dimensions, padding_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.embed(x)

        return embedding / math.sqrt(self.dimensions)
