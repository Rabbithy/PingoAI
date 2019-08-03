import time

import numpy as np


class Layer:
    def __init__(self, shape):
        self.shape = shape
        self.weights = np.random.random(shape)  * 10 ** -2
        self.biases = np.random.random(shape[0]) * 10 ** -2
        self.outputs = np.zeros(shape[0])
        self.inputs = np.zeros(shape[1])

    def activation(self, x):
        return np.tanh(x)
    
    def diff(self, y):
        return 1 - y**2
    
    def update(self, values):
        self.inputs = np.array(values)#.reshape((1, self.shape[1]))
        self.outputs = self.activation(self.weights.dot(self.inputs) + self.biases)
    
        return self.outputs

    def adjust(self, error, alpha):
        delta = error * self.diff(self.outputs)
        self.biases += delta * alpha
        self.weights += delta.reshape((self.shape[0], 1)) * self.inputs * alpha

        return delta.dot(self.weights)


'''
class BidimentionalLayer(Layer):
    def __init__(self, dim1, dim2):
        self.weights = np.random.random((dim1 + dim2))
        self.biases = np.random.random((dim1, dim2[0], 1))
        self.outputs = np.random.random((dim1, dim2[0], 1))
        self.inputs = np.random.random((dim1, 1, dim2[1]))
        self.dim1 = dim1
        self.dim2 = dim2
    
    def update(self, values):
        self.inputs = np.array(values).reshape((self.dim1, 1, self.dim2[1]))
        self.outputs = self.activation(self.weights.dot(self.inputs))
    
    def adjust(self, error, alpha):
        pass
'''


class PrimitiveMultiLayer:
    def __init__(self, layer_shape):
        print(layer_shape)
        self.layers = [Layer(layer_shape[1:]) for i in range(layer_shape[0])]
        self.shape = layer_shape
    
    def update(self, values):
        split_values = np.split(values, self.shape[0])
        for i in range(self.shape[0]):
            self.layers[i].update(split_values[i])
        
        return np.concatenate([layer.outputs for layer in self.layers])
    
    def adjust(self, error, alpha):
        s_error = np.split(error, self.shape[0])
        return np.concatenate([self.layers[i].adjust(s_error[i], alpha) for i in range(self.shape[0])])


class Network:
    def __init__(self, layers):
        self.l_shapes = layers
        self.layers = []
    
    def compile(self):
        for i in range(1, len(self.l_shapes)):
            prev = self.l_shapes[i-1]

            if type(self.l_shapes[i]) is int:
                if type(prev) is int:
                    layer = Layer((self.l_shapes[i], prev))
                else:
                    layer = Layer((self.l_shapes[i], prev[0] * prev[1]))
            else:
                if type(prev) is int:
                    shape = (self.l_shapes[i][0], self.l_shapes[i][1], prev//self.l_shapes[i][0])
                else:
                    shape = (self.l_shapes[i][0], self.l_shapes[i][1], (prev[0]*prev[1])//self.l_shapes[i][0])
                layer = PrimitiveMultiLayer(shape)
            
            # layer = Layer((self.l_shapes[i], prev))

            self.layers.append(layer)

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
