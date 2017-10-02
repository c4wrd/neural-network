import numpy as np
import random

class Neuron:

    delta = None
    output = None

    def __init__(self, num_inputs):
        self.weights = np.random.rand(num_inputs)
        self.bias = random.random()

    def forward(self, inputs):
        return self.transfer(self.activate(inputs))

    def activate(self, inputs):
        return np.dot(self.weights, inputs) + self.bias

    def transfer(self, inputs):
        return 1 / (1 + np.exp(inputs))

class Network:

    def __init__(self, num_inputs, num_layers = 1, num_nodes_layer = 3):
        self.num_layers = num_layers
        self.num_nodes_layer = num_nodes_layer
        # add our input layer
        self.input_layer = [Neuron(num_inputs) for i in range(num_nodes_layer)]
        # add our hidden layers
        self.layers = [[Neuron(num_nodes_layer) for i in range(num_nodes_layer)]]
        # add our output layer
        # TODO multiple outputs
        self.layers.append([Neuron(num_nodes_layer)])

    def forward(self, inputs):
        inputs = [node.forward(inputs) for node in self.input_layer]
        for layer in self.layers:
            new_inputs = []
            for i in range(len(layer)):
                neuron = layer[i]
                output = neuron.forward(inputs[i])
                neuron.output = output
                new_inputs.append(output)
            inputs = new_inputs
        return inputs
