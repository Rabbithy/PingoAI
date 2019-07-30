import time

import numpy as np


class Layer:
    def __init__(self, shape):
        self.shape = shape
        self.weights = np.random.random(shape) * 10 ** -10
        self.biases = np.random.random(shape[0]) * 10 ** -10
        self.outputs = np.zeros(shape[0])
        self.inputs = np.zeros(shape[1])

    def activation(self, x):
        return np.tanh(x)
        # return np.log(1 + np.exp(x))
    
    def diff(self, y):
        return 1 - y**2
        # return np.tanh(y)
    
    def update(self, values):
        self.inputs = np.array(values)
        self.outputs = self.activation(self.weights.dot(self.inputs) + self.biases)

        return self.outputs

    def adjust(self, error, alpha):
        delta = error * self.diff(self.outputs)
        self.biases += delta * alpha
        self.weights += delta.reshape((self.shape[0], 1)) * self.inputs * alpha

        return delta.dot(self.weights)


class Network:
    def __init__(self, layers):
        self.layers = [Layer((layers[i+1], layers[i])) for i in range(len(layers)-1)]

    def forward(self, values):
        for layer in self.layers:
            values = layer.update(values)
        return values
        
    def adjust(self, error, alpha):
        for layer in reversed(self.layers):
            error = layer.adjust(error, alpha)
        
    def train(self, inputs, targets, iterations, alpha):
        a = time.time()
        for i in range(iterations):
            for inp, target in zip(inputs, targets):
                out = self.forward(inp)
                error = np.array(target) - out
                self.adjust(error, alpha)
        b = time.time()
        print('time:', b - a)
