import abc

import numpy as np
import numpy.typing as npt

Tensor = npt.NDArray[np.float64]


class Module(abc.ABC):
    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...

    @abc.abstractmethod
    def backward(self, gradient: Tensor) -> Tensor:
        ...

    @abc.abstractmethod
    def step(self, lr: float) -> None:
        ...

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
