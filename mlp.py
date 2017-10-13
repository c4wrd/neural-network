import json, random
import numpy as np

from neural_network import ArtificialNeuralNetwork, Neuron
from transfer import DERIVATIVE, FUNCTION, TRANSFER_FUNCTIONS

class MLFFNetwork(ArtificialNeuralNetwork):

    input_layer = None
    layers = None

    def __init__(self, num_inputs, num_hidden_layers = 1, 
                num_nodes_layer = 3, num_outputs = 1,
                hidden_transfer = "logistic", output_transfer = "logistic"):
        """
        Constructs an instance of a multi-layer feed forward
        neural network. 

        :param num_inputs The number of inputs for this network
        :param num_hidden_layers (optional) The number of hidden layers in this network
        :param num_nodes_layer (optional) The number of units per hidden layer
        :param num_outputs (optional) The number of outputs this network produces
        :param hidden_transfer (optional) The string name of the transfer function used
            in the hidden layers
        :param output_transfer (optional) The string name of the transfer function used
            in the output layer
        """
        self.layers = []
        if num_hidden_layers > 0:
            # the first hidden layer must num_inputs inputs, whereas
            # all subsequent layers must have num_nodes_layer inputs
            self.layers.append([Neuron(num_inputs) for i in range(num_nodes_layer)])
            for i in range(num_hidden_layers - 1): # remaining layers
                self.layers.append([Neuron(num_nodes_layer) for j in range(num_nodes_layer)])
            # add our output layer
            self.layers.append([Neuron(num_nodes_layer, output_transfer) for i in range(num_outputs)])
        else:
            self.layers.append([Neuron(num_inputs, output_transfer) for i in range(num_outputs)])

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

    def train(self, data, expected_outputs, learning_rate=0.01):
        """
        Trains the neural network on an expected data point.

        :param data An array of feature values constituting one
            data point in a data set
        :param expected_outputs The expected outputs of the given
            data row
        :param learning_rate The learning rate for which weights
            will be adjusted
        """
        output = self.forward(data)
        self.backprop_error(expected_outputs)
        self.update_weights(data, learning_rate)
        return output

class PretrainedMLPNetwork(MLFFNetwork):

    def __init__(self, network_json_str):
        """
        Constructs a MLPNetwork from a pre-trained network
        output.

        :param network_json_str The json string of the pre-trained network
        """
        network_json = json.loads(network_json_str)
        output_layer = network_json["output"]
        hidden_layers = network_json["hidden"]

        # construct our layers from the 
        self.layers = [[Neuron(neuron_json=neuron) for neuron in layer] for layer in hidden_layers]
        self.layers.append([Neuron(neuron_json=neuron) for neuron in output_layer])

class MLPNetworkTrainer:

    def __init__(self, network: MLFFNetwork):
        self.network = network

    def train_linear_regression(self, dataset, num_epochs=1000, learning_rate = 0.1):
        """
        Train for function approximation (one linear output)
        """
        for epoch in range(num_epochs):
            sum_error = 0
            for row in dataset:
                inputs = row[:-1]
                expected = row[-1]
                output = self.network.train(inputs, [expected], learning_rate)[0]
                sum_error += (expected - output)**2
            print(">epoch=%d, lrate=%.3f, error=%.3f" % (epoch, learning_rate, sum_error))

    def train_classification(self, dataset, num_classes, learning_rate = 0.01, num_epochs = 1000):
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
                output = self.network.train(features, expected, learning_rate)
                sum_error = sum([(expected[i] - output[i])**2 for i in range(num_classes)])
            print('>epoch=%d, lrate=%.3f, sum_error=%.3f' % (epoch, learning_rate, sum_error))
