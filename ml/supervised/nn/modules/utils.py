from typing import Tuple, TypeVar, Union

import numpy as np

from .module import Tensor

T = TypeVar("T")
Size2d = Union[T, Tuple[T, T]]


def xavier(*shape: int, gain: float = 1.0) -> Tensor:
    bound = gain * np.sqrt(6 / np.sum(shape))
    return np.random.uniform(-bound, bound, shape)
