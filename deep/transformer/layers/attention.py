import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(
        self, heads: int, dimensions: int, dropout: float = 0.3
    ) -> None:
        self.heads = heads
        self.dimensions = dimensions

        self.head_dimensions = dimensions // heads

        self.q = nn.Linear(dimensions, dimensions)
        self.k = nn.Linear(dimensions, dimensions)
        self.v = nn.Linear(dimensions, dimensions)

        self.fc = nn.Linear(dimensions, dimensions)

        self.dropout = nn.Dropout(dropout)

        self.__weights = None

    @property
    def weights(self) -> torch.Tensor:
        if self.__weights is None:
            raise RuntimeError("No forward pass calculated.")

        return self.__weights

    def dot(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        qk = q.matmul(k.flip(2, 3)) / math.sqrt(self.head_dimensions)

        if mask is not None:
            qk = qk.masked_fill(mask.logical_not(), 1e-9)

        self.__weights = F.softmax(qk)

        return v.matmul(self.__weights)

    def forward(
        self,
        previous_q: torch.Tensor,
        previous_k: torch.Tensor,
        previous_v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = previous_q.size(0)
        shape = (batch_size, self.heads, -1, self.head_dimensions)

        q = self.q(previous_q).reshape(shape)
        k = self.k(previous_k).reshape(shape)
        v = self.v(previous_v).reshape(shape)

        out = self.dot(q, k, v, mask).reshape(
            (batch_size, -1, self.dimensions)
        )

        out = self.fc(out)

        return self.dropout(out)
