from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt


class kNearestNeighbour:
    Input = npt.NDArray[np.float64]  # No shape support in npt at the moment
    Dataset = npt.NDArray[np.float64]

    Label = np.int64
    Labels = npt.NDArray[Label]

    def __init__(
        self, k: int, dataset: Optional[Tuple[Dataset, Labels]] = None
    ) -> None:
        ...

    def fit(self, dataset: Dataset) -> None:
        """Creates a KD tree for partitioning the search space."""
        ...

    def predict(self, x: Input, dataset: Optional[Dataset] = None) -> Label:
        ...
