from typing import Optional, Tuple

import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        self.x_forget = nn.Linear(in_size, hidden_size, bias)
        self.x_input = nn.Linear(in_size, hidden_size, bias)
        self.x_output = nn.Linear(in_size, hidden_size, bias)
        self.x_cell = nn.Linear(in_size, hidden_size, bias)

        self.h_forget = nn.Linear(hidden_size, hidden_size, bias)
        self.h_input = nn.Linear(hidden_size, hidden_size, bias)
        self.h_output = nn.Linear(hidden_size, hidden_size, bias)
        self.h_cell = nn.Linear(hidden_size, hidden_size, bias)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden is None or cell is None:
            hidden = torch.zeros((x.size(0), self.hidden_size))
            cell = torch.zeros((x.size(0), self.hidden_size))

        f = torch.sigmoid(self.x_forget(x) + self.h_forget(hidden))
        i = torch.sigmoid(self.x_input(x) + self.h_input(hidden))
        o = torch.sigmoid(self.x_output(x) + self.h_output(hidden))
        c = torch.tanh(self.x_cell(x) + self.h_cell(hidden))

        updated_cell = (f * cell) + (i * c)
        updated_hidden = o * torch.tanh(updated_cell)

        return updated_hidden, updated_cell
