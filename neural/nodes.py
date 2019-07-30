import numpy as np


def activation(x):
    return 2/(1 + np.exp(-x)) - 1


def diff(y):
    return 1 - y**2


class Node:
    connections = 0     # +
    def __init__(self):
        self.connections = []
        self.bias = np.random.random()
        self.output = 0.0
        
    def connect(self, target):
        # conn = Connection(self) # -
        # target.connections.append(conn) # -
        conn = [self, target, np.random.random()] # +
        target.connections.append(conn) # +
        Node.connections += 1   # +
        
        
    def update(self):
        #z = self.bias # -
        z = self.bias + sum((conn[0].output * conn[2] for conn in self.connections)) # +
        # for conn in self.connections: # -
             # z += conn.origin.output * conn.weight # -
             # z += conn[0].output * conn[2] # + # -
        self.output = activation(z)
        
    def adjust(self, error, alpha=0.1):
        delta = diff(self.output) * error
        self.bias += delta * alpha
        for conn in self.connections:
            #conn.adjust(delta, alpha) # -
            conn[2] += delta * alpha * conn[0].output # +
            conn[0].adjust(delta * conn[2], alpha) # +

'''
class Connection:
    def __init__(self, origin):
        self.origin = origin
        self.weight = np.random.random()

    def adjust(self, error, alpha):
        self.weight += error * alpha * self.origin.output
        self.origin.adjust(error * self.weight, alpha)
'''
