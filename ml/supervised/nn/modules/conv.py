from typing import Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .module import Module, Tensor
from .utils import Size2d


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2d[int],
        stride: Size2d[int],
        padding: Size2d[int],
        bias: bool = True,
    ) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.W = np.ones((out_channels, in_channels, *kernel_size))
        self.b = np.zeros(out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.__x: Optional[Tensor] = None
        self.__grad: Optional[Tuple[Tensor, Tensor]] = None

    def __pad(self, x: Tensor) -> Tensor:
        padding = tuple((p,) for p in (0, 0, *self.padding))
        return np.pad(x, padding, "constant", constant_values=0)

    def unfold(self, x: Tensor) -> Tensor:
        stride_height, stride_width = self.stride

        # n x c_in x h_in x w_in
        # -> n x c_in x (h_in + 2p) x (w_in + 2p)
        # -> n x c_in x (h_in + 2p - (k-1)) x (w_in + 2p - (k-1)) x k x k
        # -> n x c_in x h_out x w_out x k x k
        return sliding_window_view(
            self.__pad(x), self.kernel_size, axis=(2, 3)  # type: ignore
        )[:, :, ::stride_height, ::stride_width]

    def upsample(self, x: Tensor) -> Tensor:
        stride_height, stride_width = self.stride
        x = x.repeat(stride_height, axis=2).repeat(stride_width, axis=3)

        return self.__pad(x)

    def forward(self, x: Tensor) -> Tensor:
        # n x c_in x h_in x w_in
        # -> n x c_in x h_out x w_out x k x k
        # -> n x h_out x w_out x c_in x k x k
        x = self.unfold(x).transpose(0, 2, 3, 1, 4, 5)

        # n x h_out x w_out x c_in x K
        self.__x = x.reshape((*x.shape[:4], -1))

        # n x h_out x w_out x (c_in * K)
        x = x.reshape((*x.shape[:3], -1))

        # c_out x (c_in * K)
        # -> (c_in * K) x c_out
        w = self.W.reshape((self.out_channels, -1)).T

        # (n x h_out x w_out x (c_in * K)) x ((c_in * K) x c_out)
        # -> (n x h_out x w_out x c_out)
        # -> (n x c_out x h_out x w_out)
        out = (x @ w).transpose(0, 3, 1, 2)

        if self.bias:
            out += self.b.reshape((-1, self.out_channels))

        return out

    def grad_w(self, dz: Tensor) -> Tensor:
        if self.__x is None:
            raise RuntimeError("No forward pass calculated.")

        # n x h_out x w_out x (c_in * K)
        # -> n x h_out x w_out x c_in x K
        x = self.__x.reshape((*self.__x.shape[:3], self.in_channels, -1))

        # n x h_out x w_out x c_in x K
        # -> K x c_in x n x h_out x w_out
        # -> K x c_in x N
        x = x.transpose(4, 3, 0, 1, 2).reshape(
            (x.shape[4], self.in_channels, -1)
        )

        # n x c_out x h_out x w_out
        # -> n x h_out x w_out x c_out
        # -> N x c_out
        dz = dz.transpose(0, 2, 3, 1).reshape((-1, self.out_channels))

        # (K x c_in x N) x (N x c_out) -> K x c_in x c_out
        # -> c_out x c_in x K
        # -> c_out x c_in x k x k
        return (
            (x @ dz)
            .swapaxes(0, 2)
            .reshape((self.out_channels, self.in_channels, *self.kernel_size))
        )

    def grad_b(self, dz: Tensor) -> Tensor:
        # n x c_out x h_out x w_out
        return (
            dz.transpose(1, 0, 2, 3)  # c_out x n x h_out x w_out
            .reshape((self.out_channels, -1))  # c_out x N
            .sum(axis=1)  # c_out
        )

    def grad_x(self, dz: Tensor) -> Tensor:
        # n x c_out x h_out x w_out
        # -> n x c_out x h_in x w_in
        # -> n x h_in x w_in x c_out
        dz = self.upsample(dz).transpose(0, 2, 3, 1)
        # n x h_in x w_in
        n_shape = dz.shape[:3]

        # N x c_out
        dz = dz.reshape((-1, self.out_channels))
        # c_out x c_in x K
        w = self.W.reshape((self.out_channels, self.in_channels, -1))

        # (N x c_out) x (c_out x c_in x K) -> N x c_in x K
        # -> n x h_in x w_in x c_in x K
        # -> n x h_in x w_in x c_in
        # -> n x c_in x h_in x w_in
        return (
            (dz @ w)
            .reshape((*n_shape, self.in_channels, -1))
            .sum(axis=4)
            .transpose(0, 3, 1, 2)
        )

    def backward(self, dz: Tensor) -> Tensor:
        grad_w = self.grad_w(dz)
        grad_b = self.grad_b(dz)

        self.__grad = (grad_w, grad_b)

        return self.grad_x(dz)

    def step(self, lr: float) -> None:
        if self.__grad is None:
            raise RuntimeError("No backward pass calculated.")

        grad_w, grad_b = self.__grad

        self.W -= lr * grad_w
        self.b -= lr * grad_b


if __name__ == "__main__":
    conv = Conv2d(1, 1, 3, 1, 0)
    conv.W = np.array([[[[0, 1, 0], [0, 0, 0], [0, 0, 0]]]])

    image = np.array(
        [
            [
                [
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ]
        ],
        dtype=np.float64,
    )
    out = conv(image)
    print(out)

    back = conv.backward(np.array([[[[0, 1, 0], [0, 0, 0], [0, 0, 0]]]]))
    # back = conv.backward(np.array([[[[0, 1], [0, 0]]]]))

    print(back.shape)
