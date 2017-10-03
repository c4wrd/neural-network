import numpy as np
import random

from transfer import TRANSFER_FUNCTIONS, FUNCTION, DERIVATIVE

def logistic(value):
    return 1 / (1 + np.exp(-value))

class Neuron:

    delta = None
    output = None

    def __init__(self, num_inputs, transfer_function = "logistic"):
        self.weights = np.random.rand(num_inputs)
        self.bias = random.random()
        if not transfer_function in TRANSFER_FUNCTIONS:
            raise "Invalid transfer function %s" % transfer_function
        self.transfer_fx = TRANSFER_FUNCTIONS[transfer_function]
        self.transfer_fx_name = transfer_function

    def forward(self, inputs):
        return self.transfer(self.activate(inputs))

    def activate(self, inputs):
        return np.dot(self.weights, inputs) + self.bias

    def transfer(self, activation):
        return self.transfer_fx[FUNCTION](activation)

    def transfer_derivative(self, value):
        """
        Logistic function derivative
        """
        return self.transfer_fx[DERIVATIVE](value)

    def __str__(self):
        return str(dict(
            weights = self.weights,
            bias = self.bias,
            transfer = self.transfer_fx_name
        ))

class Network:

    input_layer = None
    layers = None

    def __init__(self, num_inputs, num_layers = 1, 
                num_nodes_layer = 3, num_outputs = 1,
                hidden_transfer = "logistic", output_transfer = "logistic"):
        self.num_layers = num_layers
        self.num_nodes_layer = num_nodes_layer
        # add our input layer
        self.input_layer = [Neuron(num_inputs) for i in range(num_nodes_layer)]
        # add our hidden layers
        self.layers = [[Neuron(num_nodes_layer, hidden_transfer) for i in range(num_nodes_layer)]]
        # add our output layer
        self.layers.append([Neuron(num_nodes_layer, output_transfer) for i in range(num_outputs)])

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
            neuron.delta = error * neuron.transfer_derivative(neuron.output)

    def backprop_hidden_layers(self):
        # for each neuron in our output layer
        for i in reversed(range(len(self.layers) - 1)):
            layer = self.layers[i]
            errors = []
            for j in range(len(layer)):
                error = 0.0
                neuron = layer[j]
                ds_layer = self.layers[i + 1]
                for ds_neuron in ds_layer:
                    error += ds_neuron.weights[j] * ds_neuron.delta
                neuron.delta = error * neuron.transfer_derivative(neuron.output)

    def backprop_error(self, expected_outputs):
        self.backprop_output_layer(expected_outputs)
        self.backprop_hidden_layers()

    def update_weights(self, row, learning_rate):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            inputs = row
            if i != 0:
                prev_layer = self.layers[i-1]
                inputs = [neuron.output for neuron in prev_layer]
            for neuron in layer:
                for j in range(len(inputs)):
                    # update the jth weight for the jth input
                    neuron.weights[j] += learning_rate * neuron.delta * inputs[j]
                neuron.bias += learning_rate * neuron.delta

    def train_regression(self, dataset, learning_rate = 0.01, num_epochs = 100):
        """
        Train for function approximation (one linear output)
        """
        for epoch in range(num_epochs):
            sum_error = 0
            for row in dataset:
                inputs = row[:-1]
                expected = row[-1]
                output = self.forward(inputs)[0]
                sum_error += (expected - output)**2
                self.backprop_error([expected])
                self.update_weights(inputs, learning_rate)
            print(">epoch=%d, lrate=%.3f, error=%.3f" % (epoch, learning_rate, sum_error))

    def train_classifier(self, dataset, num_classes, learning_rate = 0.01, num_epochs = 1000):
        """
        Trains the network on the dataset, given the dataset is in the format
        [
            [x1, x2, ..., xn, C],
            ...
        ]
        Where x1..xn are features and C is a class label (0 indexed), where num_classes is the
        total number of classes.
        """
        for epoch in range(num_epochs):
            sum_error = 0
            for row in dataset:
                features = row[:-1]
                expected = [0 for i in range(num_classes)]
                expected_class = row[-1]
                expected[expected_class] = 1
                output = self.forward(features)
                sum_error = sum([(expected[i] - output[i])**2 for i in range(num_classes)])
                self.backprop_error(expected)
                self.update_weights(features, learning_rate)
            print('>epoch=%d, lrate=%.3f, sum_error=%.3f' % (epoch, learning_rate, sum_error))