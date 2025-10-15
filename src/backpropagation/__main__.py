import numpy as np

from backpropagation.network import SimpleNetwork

w1 = np.array([[1.0, 0.5], [2.0, -0.5]])
w2 = np.array([[0.5], [0.25]])

x = np.array([[0.2, 0.3], [0.4, 0.5]])
y = np.array([2, 2.1])


def main():
    network = SimpleNetwork()
    layer_outputs = network.forward(x, y)
    print("Layer outputs:", layer_outputs)


if __name__ == "__main__":
    main()
