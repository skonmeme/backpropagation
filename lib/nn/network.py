from typing import List, Optional

import numpy as np
from function import Constant, Function, ReLU, Sigmoid


class SimpleLayer:
    def __init__(
        self,
        n_input,
        n_output,
        weights: Optional[np.ndarray] = None,
        function: Function = Constant,
    ):
        self.weights = (
            weights if weights is not None else np.random.randn(n_input, n_output)
        )
        self.function = function

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.function.forward(np.dot(inputs, self.weights))

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        # fix me
        return np.dot(gradients, self.function.backward())

    def weights(self) -> np.ndarray:
        return self.weights


class SimpleNetwork:
    def __init__(self):
        self.layers = []
        self.layers.append(SimpleLayer(2, 2, np.array([0, 0, 0, 0]), ReLU()))
        self.layers.append(SimpleLayer(2, 2, np.array([0, 0]), Sigmoid()))

    def _loss(self, expected: np.ndarray, actual: np.ndarray) -> float:
        self.loss = np.sum((expected - actual) ** 2) / len(expected)
        return self.loss

    def forward(self, inputs: np.ndarray, outputs: np.ndarray) -> List[np.ndarray]:
        layer_outputs = []
        o = inputs
        for layer in self.layers:
            o = layer.forward(o)
            layer_outputs.append = o
        layer_outputs.append(self._loss(outputs, o))
        return layer_outputs

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            gradients = layer.backward(gradients)
        return gradients
