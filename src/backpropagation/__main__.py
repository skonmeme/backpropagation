import numpy as np

from backpropagation.network import SimpleNetwork

w1 = np.array([[1.0, 0.5], [2.0, -1.0]])
w2 = np.array([[0.5, 0.25]])

x = np.array([[0.2, 0.3], [0.5, 0.5]])
y = np.array([[0.5], [0.75]])


def main():
    network = SimpleNetwork(w1, w2)
    layer_outputs = network.forward(x, y)
    print("Layer outputs:", layer_outputs)
    gradients, weights = network.backward()
    print("Gradients:", gradients)
    print("Weights:", weights)


if __name__ == "__main__":
    main()
