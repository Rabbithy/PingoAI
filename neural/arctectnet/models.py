import numpy as np
from neural.optimized import networks


class StaticLayer:
    def __init__(self, size):
        self.inputs = None
        self.output = np.zeros((1, size))
        self.sync_id = None
        self.size = size
    
    @staticmethod
    def update():
        pass

    @staticmethod
    def is_sync():
        return False


class Layer(networks.Layer):
    def __init__(self, shape):
        super().__init__(shape)
        self.backlayers = []
        self.sync_id = object()
        self.size = shape[0]

    def is_sync(self):
        for layer in self.backlayers:
            if layer.is_sync() and self.sync_id != layer.sync_id:
                return False
        return True
    
    def sync(self):
        for layer in self.backlayers:
            if layer.is_sync():
                self.sync_id = layer.sync_id
                return 
    
    def get_backlayers_values(self):
        values = []
        
        if not self.is_sync():
            for layer in self.backlayers:
                if self.sync_id != layer.sync_id:
                    layer.update()
                    self.sync_id = layer.sync_id
                values.extend(layer.outputs)
        else:    
            for layer in self.backlayers:
                values.extend(layer.outputs)
        print(values)
        return np.array(values)
    
    def connect(self, layer):
        if layer not in self.backlayers:
            self.backlayers.append(layer)

    def update(self):
        if len(self.backlayers) == 0:
            raise ValueError('no input data.')
        
        values = self.get_backlayers_values()
        super().update(values)


class Network(networks.Network):
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, values):
        self.layers[0].outputs = values
        self.layers[0].sync_id = object.__new__(object)
        return self.layers[-1].update()


class ModelLayer:
    def __init__(self, size, static=False):
        self.size = size
        self.static = static
        self.connections = []
        self.comp_model = None
    
    @property
    def shape(self):
        num_connections = 0
        if len(self.connections) > 0:
            for conn in self.connections:
                num_connections += conn.size
        
        print(self.size, num_connections)
        return (self.size, num_connections)
    
    def connect(self, layer):
        self.connections.append(layer)

    def compile(self):
        if self.comp_model is None:
            if not self.static:
                layers = []
                for model_layer in self.connections:
                    layers.append(model_layer.compile())

                comp_layer = Layer(self.shape)
                for layer in layers:
                    comp_layer.connect(layer)
            else:
                comp_layer = StaticLayer(self.size)
            self.comp_model = comp_layer

        return self.comp_model


class ModelNetwork:
    def __init__(self):
        self.model_layers = []
    
    def add_model(self, model_layer):
        self.model_layers.append(model_layer)

    def remove(self, layer):
        self.model_layers.remove(layer)
    
    def compile(self):
        layers = []
        for model_layer in self.model_layers:
            layer = model_layer.compile()
            layers.append(layer)
        
        return Network(layers)
