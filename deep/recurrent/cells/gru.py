import math
from typing import Optional

import torch
import torch.nn as nn


class GRUCell(nn.Module):
    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        self.x_update = nn.Linear(in_size, hidden_size, bias)
        self.x_reset = nn.Linear(in_size, hidden_size, bias)
        self.x_hidden = nn.Linear(in_size, hidden_size, bias)

        self.h_update = nn.Linear(hidden_size, hidden_size, bias)
        self.h_reset = nn.Linear(hidden_size, hidden_size, bias)
        self.h_hidden = nn.Linear(hidden_size, hidden_size, bias)

        self.initialise()

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden is None:
            hidden = torch.zeros(
                (x.size(0), self.hidden_size), requires_grad=False
            )

        z = torch.sigmoid(self.x_update(x) + self.h_update(hidden))
        r = torch.sigmoid(self.x_reset(x) + self.h_reset(hidden))
        h = torch.tanh(r * (self.x_hidden(x) + self.h_hidden(hidden)))

        return (z * h) + ((1 - z) * hidden)

    def initialise(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
