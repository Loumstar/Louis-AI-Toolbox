import math
from typing import Optional

import numpy as np

from .conv import Conv2d
from .module import Tensor
from .utils import Size2d


class AvgPool(Conv2d):
    def __init__(
        self,
        out_channels: int,
        kernel_size: Size2d[int],
        stride: Size2d[int],
        padding: Size2d[int],
    ) -> None:
        super().__init__(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        )

        self.W = np.ones_like(self.W) / math.prod(self.kernel_size)

    def backward(self, dz: Tensor) -> Tensor:
        return self.grad_x(dz)

    def step(self, lr: float) -> None:
        pass


class MaxPool(Conv2d):
    def __init__(
        self,
        out_channels: int,
        kernel_size: Size2d[int],
        stride: Size2d[int],
        padding: Size2d[int],
    ) -> None:
        super().__init__(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        )

        self.__mask: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        # n x c x h_in x w_in
        # -> n x c x h_out x w_out x k x k
        x = self.unfold(x).reshape((*x.shape[:4], -1))
        max_x = np.amax(x, axis=4)

        self.__mask = max_x == x

        return max_x

    def backward(self, dz: Tensor) -> Tensor:
        if self.__mask is None:
            raise RuntimeError("No forward pass calculated")

        # n x c x h_out x w_out x K
        grad_x = np.zeros_like(self.__mask)
        grad_x[self.__mask] = dz

        return grad_x

    def step(self, lr: float) -> None:
        pass
