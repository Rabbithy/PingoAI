from . import nodes


class Layer:
    def __init__(self, size):
        self.nodes = [nodes.Node() for i in range(size)]

    @property
    def values(self):
        return [node.output for node in self.nodes]

    @values.setter
    def values(self, values):
        for node, value in zip(self.nodes, values):
            node.output = value
    
    def connect(self, target):
        for node in self.nodes:
            for tnode in target.nodes:
                node.connect(tnode)

    def update(self):
        for node in self.nodes:
            node.update()

    def adjust(self, error, alpha):
        for node, e in zip(self.nodes, error):
            node.adjust(e, alpha)
