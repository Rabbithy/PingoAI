import time

import numpy as np

from . import layers


class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, size):
        layer = layers.Layer(size)
        self.layers.append(layer)

    def compile(self):
        for l in range(len(self.layers)-1):
            self.layers[l].connect(self.layers[l + 1])
    
    def forward(self, values):
        self.layers[0].values = values
        for layer in self.layers[1:]:
            layer.update()
        return self.layers[-1].values
    
    def backprop(self, error, alpha):
        self.layers[-1].adjust(error, alpha)

    def train(self, inputs, outputs, iterations, alpha):
        a = time.time()
        for i in range(iterations):
            for inp, out in zip(inputs, outputs):
                error = np.array(out) - self.forward(inp)
                self.backprop(error, alpha)
        b = time.time()
        print('time:', b - a)


 
