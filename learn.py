import numpy as np
import random

def logistic(value):
    return 1 / (1 + np.exp(-value))

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

    def transfer(self, activation):
        """
        Logistic function transfer
        """
        return 1 / (1 + np.exp(-activation))

    def __str__(self):
        return str(dict(
            weights = self.weights,
            bias = self.bias
        ))

class Network:

    input_layer = None
    layers = None

    def __init__(self, num_inputs, num_layers = 1, num_nodes_layer = 3, num_outputs = 1):
        self.num_layers = num_layers
        self.num_nodes_layer = num_nodes_layer
        # add our input layer
        self.input_layer = [Neuron(num_inputs) for i in range(num_nodes_layer)]
        # add our hidden layers
        self.layers = [[Neuron(num_nodes_layer) for i in range(num_nodes_layer)]]
        # add our output layer
        self.layers.append([Neuron(num_nodes_layer) for i in range(num_outputs)])

    def forward(self, inputs):
        inputs = [node.forward(inputs) for node in self.input_layer]
        for layer in self.layers:
            new_inputs = []
            for neuron in layer:
                output = neuron.forward(inputs)
                neuron.output = output
                new_inputs.append(output)
            inputs = new_inputs
        return inputs

    def backprop_output_layer(self, expected_outputs):
        layer = self.layers[-1]
        errors = []
        for j in range(len(layer)):
            neuron = layer[j]
            error = expected_outputs[j] - neuron.output
            neuron.delta = error * self.logistic_derivative(neuron.output)

    def backprop_hidden_layers(self):
        # for each neuron in our output layer
        for i in reversed(range(len(self.layers) - 1)):
            layer = self.layers[i]
            errors = []
            for j in range(len(layer)):
                neuron = layer[j]
                error = 0.0
                for ds_neuron in self.layers[i + 1]:
                    error += ds_neuron.weights[j] * ds_neuron.delta
                neuron.delta = error * self.logistic_derivative(neuron.output)

    def backprop_error(self, expected_outputs):
        self.backprop_output_layer(expected_outputs)
        self.backprop_hidden_layers()

    def update_weights(self, row, learning_rate):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            inputs = row # remove class identifier from this data input
            if i != 0:
                prev_layer = self.layers[i-1]
                inputs = [neuron.output for neuron in prev_layer]
            for neuron in layer:
                for j in range(len(inputs)):
                    # update the jth weight for the jth input
                    neuron.weights[j] += learning_rate * neuron.delta * inputs[j]
                neuron.bias += learning_rate * neuron.delta

    def logistic_derivative(self, output_of_neuron):
        return output_of_neuron * (1.0 - output_of_neuron)

    def train(self, dataset, n_outputs, learning_rate = 0.01, num_epochs = 100):
        """
        Train for function approximation (one linear output)
        """
        for epoch in range(num_epochs):
            sum_error = 0
            for row in dataset:
                inputs = row[:-1]
                expected = logistic(row[-1])
                output = self.forward(inputs)[0]
                print("expected: %s, got: %s" % (expected, output))
                sum_error += (expected - output)**2
                self.backprop_error([expected])
                self.update_weights(inputs, learning_rate)
            print(">epoch=%d, lrate=%.3f, error=%.3f" % (epoch, learning_rate, sum_error))