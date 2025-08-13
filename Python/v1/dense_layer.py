from typing import Callable
import numpy as np

# type declarations, allow for configurable precision
int_t = np.int32
float_t = np.float32

class Activation:
    def __init__(self, function: Callable, derivative: Callable):
        self.function: Callable = np.vectorize(function, otypes=[float_t])
        self.derivative: Callable = np.vectorize(derivative, otypes=[float_t])
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.function(x)
    def differentiate(self, x: np.ndarray) -> np.ndarray:
        return self.derivative(x)

class LinearActivation(Activation):
    def __init__(self):
        super().__init__(lambda x: x, lambda x: 1)

class ReLUActivation(Activation):
    def __init__(self):
        super().__init__(lambda x: np.maximum(x, 0), lambda x: 0 if x < 0 else 1)

class DenseLayer:
    def __init__(self, size: int_t, activation: Activation):
        self.size: int_t = size
        self.activation: Activation = activation
        self.values: np.ndarray = np.empty(size, dtype=float_t)
    def load(self, x: np.ndarray) -> None:
        self.values[:] = x
    def activated(self) -> np.ndarray:
        return self.activation.evaluate(self.values)
    def __repr__(self) -> str:
        return f'{str(self.values)} -> {self.activated()}'

class DenseInterconnect:
    def __init__(self, in_size: int_t, out_size: int_t):
        self.size: tuple[int_t, int_t] = (in_size, out_size)
        self.weights: np.ndarray = float_t(2) * np.random.random_sample(self.size).astype(float_t) - float_t(1)
        self.biases: np.ndarray = float_t(2) * np.random.random_sample(self.size[1]).astype(float_t) - float_t(1)
    def forward_pass(self, x: np.ndarray, y: np.ndarray) -> None:
        np.matmul(x, self.weights, out=y)
        np.add(y, self.biases, out=y)
    def __repr__(self) -> str:
        return f'{str(self.weights)} + {str(self.biases)}'

class DenseNetwork:
    def __init__(self, *dense_layers: DenseLayer):
        self.dense_layers: list[DenseLayer] = []
        self.dense_interconnects: list[DenseInterconnect] = []

        previous_size = dense_layers[0].size
        self.dense_layers.append(dense_layers[0])
        for dense_layer in dense_layers[1:]:
            current_size = dense_layer.size
            interconnect = DenseInterconnect(previous_size, current_size)
            previous_size = current_size

            self.dense_layers.append(dense_layer)
            self.dense_interconnects.append(interconnect)
    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        self.dense_layers[0].load(x)
        for idx, interconnect in enumerate(self.dense_interconnects):
            from_layer = self.dense_layers[idx]
            to_layer = self.dense_layers[idx + 1]
            interconnect.forward_pass(from_layer.activated(), to_layer.values)
        return self.dense_layers[-1].activated()
    def __repr__(self) -> str:
        reprs: list[str] = [repr(self.dense_layers[0])]
        for idx, interconnect in enumerate(self.dense_interconnects):
            reprs.append(repr(interconnect))
            reprs.append(repr(self.dense_layers[idx + 1]))
        return '\n'.join(reprs)

if __name__ == '__main__':
    l1 = DenseLayer(int_t(2), ReLUActivation())
    l2 = DenseLayer(int_t(2), ReLUActivation())
    l3 = DenseLayer(int_t(2), ReLUActivation())

    n = DenseNetwork(l1, l2, l3)
    n.forward_pass(np.array([1, 2]))
    print(repr(n))