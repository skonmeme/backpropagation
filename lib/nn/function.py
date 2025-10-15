from typing import Union

import numpy as np


class ReLU:
    values: np.ndarray = None

    def forward(self, x: Union[float, np.ndarray]) -> np.ndarray:
        if isinstance(x, np.ndarray):
            self.values = x
        else:
            self.values = np.array([x])
        return np.maximum(0, self.values)

    def backward(self) -> np.ndarray:
        if self.values is None:
            raise ValueError("ReLU has not been called yet")
        return np.where(self.values > 0, 1, 0)


class Sigmoid:
    values: np.ndarray = None

    def forward(self, x: Union[float, np.ndarray]) -> np.ndarray:
        if isinstance(x, np.ndarray):
            self.values = x
        else:
            self.values = np.array([x])
        return 1 / (1 + np.exp(-self.values))

    def backward(self) -> np.ndarray:
        if self.derivatives is None:
            raise ValueError("Sigmoid has not been called yet")
        sigmoid = 1 / (1 + np.exp(-self.values))
        return sigmoid * (1 - sigmoid)
