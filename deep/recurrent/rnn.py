from typing import Type

import torch
import torch.nn as nn

from . import cells


class UniDirectionalRNN(nn.Module):
    def __init__(
        self,
        cell_type: Type[cells.CellType],
        in_size: int,
        out_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
        **kwargs
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.cells = nn.ModuleList()

        for i in range(num_layers):
            in_features = in_size if i == 0 else hidden_size
            cell = cell_type(in_features, hidden_size, bias, **kwargs)

            self.cells.append(cell)

        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hiddens = []
        outs = []

        for step in range(x.size(1)):
            x_step = x[:, step]
            layer_hiddens = []

            for j, layer in enumerate(self.cells):
                if isinstance(layer, cells.LSTMCell):
                    hidden = hiddens[step - 1][j] if step > 0 else (None, None)
                    x_step = layer(x_step, *hidden)
                else:
                    hidden = hiddens[step - 1][j] if step > 0 else None
                    x_step = layer(x_step, hidden)

                layer_hiddens.append(x_step)

                if isinstance(layer, cells.LSTMCell):
                    x_step = x_step[0]

                if j == self.num_layers - 1:
                    outs.append(x_step)

            hiddens.append(layer_hiddens)

        return self.fc(outs[-1].squeeze())


class BiDirectionalRNN(nn.Module):
    def __init__(
        self,
        cell_type: Type[cells.CellType],
        in_size: int,
        out_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
        **kwargs
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.cells = nn.ModuleList()

        for i in range(num_layers):
            in_features = in_size if i == 0 else hidden_size
            cell = cell_type(in_features, hidden_size, bias, **kwargs)

            self.cells.append(cell)

        self.fc = nn.Linear(2 * hidden_size, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forward_hiddens = []
        reverse_hiddens = []

        forward_outs = []
        reverse_outs = []

        for step in range(x.size(1)):
            x_forward, x_reverse = x[:, step], x[:, -step]

            forward_layer_hiddens = []
            reverse_layer_hiddens = []

            for j, cell in enumerate(self.cells):
                if isinstance(cell, cells.LSTMCell):
                    if step > 0:
                        forward_hidden = forward_hiddens[step - 1][j]
                        reverse_hidden = reverse_hiddens[step - 1][j]
                    else:
                        forward_hidden = reverse_hidden = (None, None)

                    x_forward = cell(x_forward, *forward_hidden)
                    x_reverse = cell(x_reverse, *reverse_hidden)

                else:
                    if step > 0:
                        forward_hidden = forward_hiddens[step - 1][j]
                        reverse_hidden = reverse_hiddens[step - 1][j]
                    else:
                        forward_hidden = reverse_hidden = None

                    x_forward = cell(x_forward, forward_hidden)
                    x_reverse = cell(x_reverse, reverse_hidden)

                forward_layer_hiddens.append(x_forward)
                reverse_layer_hiddens.append(x_reverse)

                if isinstance(cell, cells.LSTMCell):
                    x_forward, x_reverse = x_forward[0], x_reverse[0]

                if j == self.num_layers - 1:
                    forward_outs.append(x_forward)
                    reverse_outs.append(x_reverse)

            forward_hiddens.append(forward_layer_hiddens)
            reverse_hiddens.append(reverse_layer_hiddens)

        forward_out = forward_outs[-1].squeeze()
        reverse_out = reverse_outs[-1].squeeze()

        out = torch.cat((forward_out, reverse_out), 1)

        return self.fc(out)
