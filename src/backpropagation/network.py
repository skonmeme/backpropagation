from typing import List, Optional

import numpy as np

from backpropagation.function import Constant, Function, ReLU, Sigmoid


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
        self.inputs = inputs
        self.outputs = self.function.forward(self.weights @ inputs)
        return self.outputs

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        # fix me
        gradients = self.function.backward(gradients)
        self.weights -= self.learning_rate * gradients @ self.inputs.T
        return gradients @ self.weights.T


class SimpleNetwork:
    def __init__(
        self, w1: np.ndarray, w2: np.ndarray, leanring_rate: np.float32 = 0.05
    ):
        self.layers = []
        self.layers.append(SimpleLayer(2, 2, w1, ReLU()))
        self.layers.append(SimpleLayer(2, 1, w2, Sigmoid()))
        self.learning_rate = leanring_rate

    def _loss(self, expected: np.ndarray, actual: np.ndarray) -> float:
        self.loss = np.sum((expected.transpose() - actual) ** 2)
        return self.loss

    def forward(self, inputs: np.ndarray, outputs: np.ndarray) -> List[np.ndarray]:
        self.inputs = inputs
        self.outputs = outputs
        layer_outputs = []
        o = inputs
        for layer in self.layers:
            o = layer.forward(o)
            layer_outputs.append(o)
            self.forwared = o
        layer_outputs.append(self._loss(outputs, o))
        return [layer_outputs, self._loss(outputs, o)]

    def backward(self) -> (List[np.ndarray], List[np.ndarray]):
        gradient = 2 * (self.forwared - self.outputs)
        gradients = [gradient]
        weights = []
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
            gradients.append(gradient)
            weights.append(layer.weights)
        return (gradients, weights)
