from typing import Literal, Optional

import torch
import torch.nn as nn


class VanillaCell(nn.Module):
    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        bias: bool = True,
        activation: Literal["tanh", "relu"] = "tanh",
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        self.x = nn.Linear(in_size, hidden_size, bias)
        self.h = nn.Linear(hidden_size, hidden_size, bias)
        self.activation = nn.Tanh() if activation == "tanh" else nn.ReLU()

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if hidden is None:
            hidden = torch.zeros((x.size(0), self.hidden_size))

        out = self.x(x) + self.h(hidden)
        return self.activation(out)
