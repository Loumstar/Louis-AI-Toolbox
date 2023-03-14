from __future__ import annotations

import dataclasses
from typing import Callable, Dict, Literal, Optional, Union

import numpy as np
import numpy.typing as npt

Input = npt.NDArray[np.float64]  # No shape support in npt at the moment
Dataset = npt.NDArray[np.float64]

Label = np.int64
Labels = npt.NDArray[Label]

DistanceFunction = Callable[[Input, Input], np.float64]


@dataclasses.dataclass
class Node:
    data: Input
    label: Label
    left: Optional[Node] = None
    right: Optional[Node] = None


class kNearestNeighbour:
    SupportedFunctions = Literal["manhatten", "euclidean"]

    DISTANCE_FUNCTIONS: Dict[SupportedFunctions, DistanceFunction] = {
        "euclidean": lambda x, y: np.sum((x - y) ** 2),
        "manhatten": lambda x, y: np.sum(np.abs(x - y)),
    }

    def __init__(
        self,
        k: int,
        dataset: Optional[Dataset] = None,
        labels: Optional[Labels] = None,
        distance: Union[SupportedFunctions, DistanceFunction] = "euclidean",
    ) -> None:
        self.k = k
        self.tree: Optional[Node] = None

        self.distance = (
            distance
            if distance not in self.DISTANCE_FUNCTIONS
            else self.DISTANCE_FUNCTIONS[distance]
        )

        if dataset is not None and labels is not None:
            self.fit(dataset, labels)

    @property
    def root(self) -> Optional[Node]:
        return self.__root

    def __insert(
        self, x: Input, y: Label, node: Optional[Node] = None, d: int = 0
    ):
        if node is None:
            return Node(x, y)

        next_d = (d + 1) % x.shape[0]

        if x[d] <= node.data[d]:
            node.left = self.__insert(x, y, node.left, next_d)
        else:
            node.right = self.__insert(x, y, node.right, next_d)

    def __balance(self, tree: Optional[Node]) -> None:
        ...

    def fit(self, dataset: Dataset, labels: Labels) -> None:
        """Creates a KD tree for partitioning the search space."""

        for x, y in zip(dataset, labels):
            self.__root = self.__insert(x, y, self.root)

        self.__balance(self.root)

    def predict(
        self,
        x: Input,
        k: Optional[int] = None,
        dataset: Optional[Dataset] = None,
    ) -> Label:
        ...
