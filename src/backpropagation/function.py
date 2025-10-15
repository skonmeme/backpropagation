from typing import Union

import numpy as np


class Function:
    values: np.ndarray = None

    def forward(self, x: Union[float, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError


class Constant(Function):
    values: np.ndarray = None

    def forward(self, x: Union[float, np.ndarray]) -> np.ndarray:
        if isinstance(x, np.ndarray):
            self.values = x
        else:
            self.values = np.array([x])
        return self.values

    def backward(self) -> np.ndarray:
        if self.values is None:
            raise ValueError("Constant has not been called yet")
        return np.zeros_like(self.values)


class ReLU(Function):
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


class Sigmoid(Function):
    output: np.ndarray = None

    def forward(self, x: Union[float, np.ndarray]) -> np.ndarray:
        if isinstance(x, np.ndarray):
            values = x
        else:
            values = np.array([x])
        self.output = 1 / (1 + np.exp(-values))
        return self.output

    def backward(self) -> np.ndarray:
        if self.output is None:
            raise ValueError("Sigmoid has not been called yet")
        return self.output * (1 - self.output)
