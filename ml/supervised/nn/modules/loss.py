from typing import Optional, Tuple

import numpy as np

from .activation import Softmax
from .module import Module, Tensor


class MeanSquaredErrorLoss(Module):
    def __init__(self) -> None:
        self.__labels: Optional[Tuple[Tensor, Tensor]] = None

    def forward(self, pred: Tensor, target: Tensor) -> np.float64:
        self.__labels = (pred, target)

        return np.mean((pred - target) ** 2)

    def backward(self) -> Tensor:
        if self.__labels is None:
            raise ValueError("No forward pass saved.")

        pred, target = self.__labels

        return 2 * (pred - target) - target.shape[0]

    def step(self, lr: float) -> None:
        pass


class CrossEntropyLoss(Module):
    def __init__(self) -> None:
        self.__labels: Optional[Tuple[Tensor, Tensor]] = None
        self.softmax = Softmax()

    def forward(self, pred: Tensor, target: Tensor) -> np.float64:
        probs = self.softmax(pred)
        self.__labels = (probs, target)

        return -(1 / target.shape[0]) * np.sum(target * np.log(probs))

    def backward(self) -> Tensor:
        if self.__labels is None:
            raise ValueError("No forward pass saved.")

        probs, target = self.__labels

        return -(1 / target.shape[0]) * (target - probs)

    def step(self, lr: float) -> None:
        pass
