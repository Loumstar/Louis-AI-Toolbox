from typing import Optional, Tuple

import numpy as np

from .kernels import Kernel
from .utils import TArray


class GaussianProcessRegressor:
    def __init__(self, kernel: Kernel, eps: float = 1e-8) -> None:
        self.kernel = kernel
        self.eps = eps

        self._data: Optional[Tuple[TArray, TArray]] = None
        self._k_ff: Optional[TArray] = None
        self._k_ff_inverse: Optional[TArray] = None

    @property
    def x_train(self) -> Optional[TArray]:
        if self._data is None:
            return None

        return self._data[0]

    @property
    def y_train(self) -> Optional[TArray]:
        if self._data is None:
            return None

        return self._data[1]

    def fit(self, x: TArray, y: TArray, reset: bool = False) -> None:
        if x.shape[0] != x.shape[0]:
            raise ValueError(
                "X and Y arrays must have the same number of samples."
            )

        if reset or self._data is None:
            x_train, y_train = x, y
        else:
            if x.ndim != self._data[0] or y.ndim != self._data[1]:
                raise ValueError(
                    "New X and Y arrays must have the same number of "
                    "dimensions as the dataset."
                )

            x_train = np.append(self._data[0], x, axis=0)
            y_train = np.append(self._data[1], y, axis=0)

        self._data = (x_train, y_train)
        self._k_ff = self.kernel(x_train, x_train)

        identity = np.eye(x_train.shape[0])
        self._k_ff_inverse = np.linalg.inv(self._k_ff + self.eps * identity)

    def predict(self, x: TArray) -> Tuple[TArray, TArray]:
        if self._data is None or self._k_ff_inverse is None:
            raise ValueError("GP hasn't been fitted yet.")

        x_train, y_train = self._data

        k_yy = self.kernel(x, x)
        k_fy = self.kernel(x_train, x)

        print(k_yy.shape, k_fy.shape, self._k_ff_inverse.shape)

        mean = k_fy.T.dot(self._k_ff_inverse).dot(y_train)
        covariance = k_yy - k_fy.T.dot(self._k_ff_inverse).dot(k_fy)

        return mean, covariance


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from .kernels import Gaussian

    noise = 0.5

    x_train = np.array([3, 1, 4, 5, 9]).reshape(-1, 1)
    y_train = np.cos(x_train) + np.random.normal(0, noise, size=x_train.shape)

    x_test = np.arange(0, 10, 0.1).reshape(-1, 1)

    kernel = Gaussian(0.5, 0.2)
    gp = GaussianProcessRegressor(kernel)

    gp.fit(x_train, y_train)
    mu, cov = gp.predict(x_test)

    y_test = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.figure()
    plt.title("length=%.2f variance=%.2f" % (kernel.length, kernel.variance))

    plt.fill_between(
        x_test.ravel(),
        y_test + uncertainty,  # type: ignore
        y_test - uncertainty,  # type: ignore
        alpha=0.1,
    )

    plt.plot(x_test, y_test, label="predict")
    plt.scatter(x_train, y_train, label="train", c="red")

    plt.legend()
    plt.show()
