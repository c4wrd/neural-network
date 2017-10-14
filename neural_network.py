import json, random
import numpy as np

from transfer import DERIVATIVE, FUNCTION, TRANSFER_FUNCTIONS

class Neuron:

    delta = None
    output = None

    def __init__(self, num_inputs = None, transfer_function = "logistic",
                neuron_json = None):
        if neuron_json is None:
            self.weights = np.random.rand(num_inputs)
            self.bias = random.random()
        else:
            self.weights = neuron_json["weights"]
            self.bias = neuron_json["bias"]
            transfer_function = neuron_json["transfer"]
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

    def json(self):
        return dict(
            weights = [float(weight) for weight in self.weights],
            bias = self.bias,
            transfer = self.transfer_fx_name
        )

class ArtificialNeuralNetwork:

    input_layer = None
    layers = None

    def __init__(self):
        raise NotImplementedError("This is an abstract class and must be inherited!")

    def forward(self, inputs):
        for layer in self.layers:
            new_inputs = []
            for neuron in layer:
                output = neuron.forward(inputs)
                neuron.output = output
                new_inputs.append(output)
            inputs = new_inputs
        return inputs

    def backprop_output_layer(self, expected_outputs):
        """
        A helper method to calculate the error delta for the neurons
        in the output layer
        """
        layer = self.layers[-1]
        for j in range(len(layer)):
            neuron = layer[j]
            error = expected_outputs[j] - neuron.output
            neuron.delta = error * neuron.transfer_derivative(neuron.output)

    def backprop_hidden_layers(self):
        """
        A helper function to backprop the error and calculate the 
        deltas in each neuron in each layer of the hidden layers
        """
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
        """
        Backpropagates the error from the given expected outputs
        through the network.

        :param expected_outputs The target outputs of the network
            for which error will be calculated for
        """
        self.backprop_output_layer(expected_outputs)
        self.backprop_hidden_layers()

    def update_weights(self, row, learning_rate):
        """
        Updates the weights for all of the layers in the network.

        :param row A particular data point for which the weights will
            be adjusted for
        :param learning_rate The learning rate to apply during weight update
            calculations
        """
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

    def json(self):
        json_network = {
            "output": [neuron.json() for neuron in self.layers[-1]],
            "hidden": []
        }

        if len(self.layers) > 1:
            json_network["hidden"] = [[neuron.json() for neuron in layer] for layer in self.layers[:-1]]
        
        return json_network