from typing import Optional

import numpy as np

from .module import Module, Tensor


class ReLU(Module):
    def __init__(self) -> None:
        self.__z: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        self.__z = np.maximum(0, x)
        return self.__z

    def backward(self, gradient: Tensor) -> Tensor:
        if self.__z is None:
            raise ValueError("No forward pass saved.")

        grad_x = gradient.copy()
        grad_x[self.__z < 0] = 0

        return grad_x

    def step(self, lr: float) -> None:
        pass


class LeakyReLU(Module):
    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self.__z: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        self.__z = x.copy()
        self.__z[x < 0] = self.alpha * x

        return self.__z

    def backward(self, gradient: Tensor) -> Tensor:
        if self.__z is None:
            raise ValueError("No forward pass saved.")

        grad_x = (1 - self.alpha) * gradient.copy()
        grad_x[self.__z < 0] = 0

        return grad_x + self.alpha

    def step(self, lr: float) -> None:
        pass


class Sigmoid(Module):
    def __init__(self) -> None:
        self.__z: Optional[Tensor] = None

    @staticmethod
    def __forward(x: Tensor) -> Tensor:
        return 1 / (1 + np.exp(-x))

    def forward(self, x: Tensor) -> Tensor:
        self.__z = self.__forward(x)
        return self.__z

    def backward(self, gradient: Tensor) -> Tensor:
        if self.__z is None:
            raise ValueError("No forward pass saved.")

        s = self.__forward(self.__z)

        return gradient * s * (1 - s)

    def step(self, lr: float) -> None:
        pass


class Tanh(Module):
    def __init__(self) -> None:
        self.__z: Optional[Tensor] = None

    @staticmethod
    def __forward(x: Tensor) -> Tensor:
        return np.tanh(x)

    def forward(self, x: Tensor) -> Tensor:
        self.__z = self.__forward(x)
        return self.__z

    def backward(self, gradient: Tensor) -> Tensor:
        if self.__z is None:
            raise ValueError("No forward pass saved.")

        s = self.__forward(self.__z)

        return gradient * (1 - (s**2))

    def step(self, lr: float) -> None:
        pass


class Softmax(Module):
    def __init__(self) -> None:
        self.__z: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        axes = range(1, x.ndim)  # all but the first

        exponent = np.exp(x - np.max(x, axis=axes))
        self.__z = exponent / np.sum(exponent, axis=axes, keepdims=True)

        return self.__z

    def backward(self, gradient: Tensor) -> Tensor:
        if self.__z is None:
            raise ValueError("No forward pass saved.")

        # s = Z_i(d_ij - Z_j)
        s = -(self.__z * self.__z.T) + np.diag(self.__z)

        return gradient * s

    def step(self, lr: float) -> None:
        pass
