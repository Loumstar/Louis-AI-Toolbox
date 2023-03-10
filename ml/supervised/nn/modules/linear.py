from typing import Optional, Tuple

import numpy as np

from .module import Module, Tensor
from .utils import xavier


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.W = xavier(in_features, out_features)
        self.b = xavier(1, out_features)

        self.__x: Optional[Tensor] = None
        self.__grad: Optional[Tuple[Tensor, Tensor]] = None

    def forward(self, x: Tensor) -> Tensor:
        self.__x = x
        return (x @ self.W) + self.b

    def backward(self, gradient: Tensor) -> Tensor:
        if self.__x is None:
            raise ValueError("No forward pass saved.")

        grad_w = self.__x.T @ gradient
        grad_b = np.sum(gradient, axis=0)

        self.__grad = (grad_w, grad_b)

        return gradient @ self.W.T

    def step(self, lr: float) -> None:
        if self.__grad is None:
            raise ValueError("No backwards pass calculated.")

        grad_w, grad_b = self.__grad

        self.W -= lr * grad_w
        self.b -= lr * grad_b

        self.__grad = None
