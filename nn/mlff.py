import json
import util
import numpy as np

from dataset import Datasets
from nn.neural_network import ArtificialNeuralNetwork, Neuron

def get_mean_gradients(dweights):
    """
    Calculates the mean weight gradient for a supplied
    list of weight gradients acquired over multiple training
    updates. Because the weight gradient is of multiple dimensions,
    we must iterate over each layer, neuron and weight individually to
    calculate the average.
    :param dweights: An array of gradient weight matrices
    :returns The average weight gradients
    """
    summed_dweights = dweights.pop()
    num_dweights = len(dweights)

    # accumulate
    for layer_index in range(num_dweights):
        layer_dweights = summed_dweights[layer_index]
        for neuron_index in range(len(layer_dweights)):
            neuron_dweights = layer_dweights[neuron_index]
            for dweight_index in range(len(neuron_dweights)):
                summed_dweights[layer_index][neuron_index][dweight_index] += num_dweights + 1

    return summed_dweights
    # mean

    return summed_dweights

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
        for j in range(len(layer)): # for each neuron in the output layer
            neuron = layer[j]
            error = expected_outputs[j] - neuron.output
            neuron.delta = error * neuron.transfer_derivative(neuron.output)
        return [neuron.delta for neuron in layer]

    def backprop_hidden_layers(self):
        """
        A helper function to backprop the error and calculate the
        deltas in each neuron in each layer of the hidden layers
        """
        # for each neuron in our output layer
        for i in reversed(range(len(self.layers) - 1)):
            layer = self.layers[i]
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

        :param expected_outputs: The target outputs of the network
            for which error will be calculated for
        :returns deltas: The array of computed deltas when propagating
            the expected_outputs
        """
        self.backprop_output_layer(expected_outputs)
        self.backprop_hidden_layers()
        return [[neuron.delta for neuron in layer] for layer in self.layers]

    def calculate_weight_gradients(self, row):
        """
        Calculates the weight gradients and bias change for each weight of each neuron
            in each layer of the network without updating the weights.

        :param row: A particular data point for which the weights will
            be adjusted for
        :param learning_rate: The learning rate to apply during weight update
            calculations
        """
        weight_gradients = []
        for i in range(len(self.layers)):
            layer_gradients = []
            layer = self.layers[i]
            inputs = row
            if i != 0:
                prev_layer = self.layers[i-1]
                inputs = [neuron.output for neuron in prev_layer]
            for neuron in layer:
                neuron_gradients = []
                for j in range(len(inputs)):
                    # update the jth weight for the jth input
                    neuron_gradients.append(self.learning_rate * neuron.delta * inputs[j])
                neuron_gradients.append(self.learning_rate * neuron.delta) # store the bias change last in the array
                layer_gradients.append(neuron_gradients)
            weight_gradients.append(layer_gradients)
        return weight_gradients

    def apply_weight_gradients(self, delta_weights):
        """
        Updates the weights for each neuron in each layer according
        to the weight change vector.
        """
        for l in range(len(delta_weights)):
            layer_weights = delta_weights[l]
            for n in range(len(layer_weights)):
                neuron_weights = layer_weights[n]
                neuron = self.layers[l][n]
                for w in range(len(neuron.weights)):
                    neuron.weights[w] += neuron_weights[w] # update the weight with the weight change
                neuron.bias += neuron_weights[-1] # the bias is stored at the last weight index

    def update_weights(self, row):
        """
        Updates the weights for all of the layers in the network
            from a single data point (stochastic gradient descent).

        :param row A particular data point for which the weights will
            be adjusted for
        """
        dweights = self.calculate_weight_gradients(row)
        self.apply_weight_gradients(dweights)
        # for i in range(len(self.layers)):
        #     layer = self.layers[i]
        #     inputs = row
        #     if i != 0:
        #         prev_layer = self.layers[i-1]
        #         inputs = [neuron.output for neuron in prev_layer]
        #     for neuron in layer:
        #         for j in range(len(inputs)):
        #             # update the jth weight for the jth input
        #             neuron.weights[j] += self.learning_rate * neuron.delta * inputs[j]
        #         neuron.bias += self.learning_rate * neuron.delta

    def train(self, inputs, expected_outputs):
        """
        Trains the neural network on an expected data point.

        :param data An array of feature values constituting one
            data point in a data set
        :param expected_outputs The expected outputs of the given
            data row
        :param learning_rate The learning rate for which weights
            will be adjusted
        """
        output = self.forward(inputs)
        self.backprop_error(expected_outputs)
        self.update_weights(inputs)
        return output

    def train_without_update(self, inputs, expected_outputs):
        outputs = self.forward(inputs)
        self.backprop_error(expected_outputs)
        dweights = self.calculate_weight_gradients(inputs)
        return [outputs, dweights]

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

class MLFFNetworkTrainer:

    def __init__(self, network: MLFFNetwork):
        self.network = network

    def train_regression_stochastic(self, dataset, num_epochs=1000, start_epoch=0, learning_rate = 0.1):
        """
        Trains the network on regression, yielding the epoch and sum_error for each epoch
        """
        self.network.set_learning_rate(learning_rate)
        if num_epochs == -1:
            num_epochs = 1000000000000
        for epoch in range(start_epoch, start_epoch + num_epochs):
            sum_error = 0
            for row in dataset:
                inputs = row[:-1]
                expected = row[-1:]
                output = self.network.train(inputs, expected)[0]
                sum_error += (expected - output)**2
            yield [epoch, sum_error/2]

    def train_regression_batch(self, dataset, learning_rate=0.1, num_epochs=1000, start_epoch=0, batch_size=None):
        batches = None
        if batch_size is None:
            batches = [dataset]
        else:
            batches = util.chunk_array(dataset, batch_size)

        self.network.set_learning_rate(learning_rate)
        if num_epochs == -1:
            num_epochs = 1000000000000
        for epoch in range(start_epoch, start_epoch + num_epochs):
            sum_error = 0
            for batch in batches:
                batch_dweights = []  # array of weight changes for data points in this batch
                for row in batch:
                    inputs = row[:-1]
                    expected = row[-1:]
                    [output, dweights] = self.network.train_without_update(inputs, expected)  # calculate the outputs
                    sum_error += (expected - output) ** 2
                    self.network.backprop_error(expected)
                    batch_dweights.append(dweights)
                mean_dweights = get_mean_gradients(batch_dweights)
                self.network.apply_weight_gradients(mean_dweights)
            yield [epoch, sum_error / 2]

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
                output = self.network.train(features, expected)
                sum_error = sum([(expected[i] - output[i])**2 for i in range(num_classes)])
                yield dict(epoch=epoch, error=sum_error)
            # print('>epoch=%d, lrate=%.3f, sum_error=%.3f' % (epoch, learning_rate, sum_error))

dataset = Datasets.linear()
network = MLFFNetwork(1, 1, 50, 1, output_transfer="linear")
trainer = MLFFNetworkTrainer(network)

