from abc import ABC, abstractmethod

import numpy as np
from numpy import float64

from .utils import TArray


class Kernel(ABC):
    @abstractmethod
    def __call__(self, x1: TArray, x2: TArray) -> TArray:
        pass


class Constant(Kernel):
    def __init__(self, value: float) -> None:
        self.value = float64(value)

    def __call__(self, x1: TArray, x2: TArray) -> TArray:
        return np.full((x1.shape[0], x2.shape[0]), fill_value=self.value)


class White(Kernel):
    def __init__(self, noise: float = 1.0, atol: float = 0.01) -> None:
        self.noise = float64(noise)
        self.zero = float64(0)

        self.atol = atol

    def __call__(self, x1: TArray, x2: TArray) -> TArray:
        differences = np.abs(x1 - x2.T)

        mask = differences <= self.atol
        return self.noise * mask


class Gaussian(Kernel):
    def __init__(self, length: float, variance: float):
        super().__init__()

        self.length = length
        self.variance = variance

    def __call__(self, x1: TArray, x2: TArray) -> TArray:
        x1_abs = np.sum(x1**2, 1).reshape(-1, 1)
        x2_abs = np.sum(x2**2, 1)

        distances = x1_abs + x2_abs - 2 * np.dot(x1, x2.T)

        exponential = np.exp(-distances / (2 * self.length**2))
        return self.variance**2 * exponential
