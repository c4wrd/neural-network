import json

import numpy as np

import util
from nn.neural_network import Neuron, ArtificialNeuralNetwork

class RBFNeuron(Neuron):

    def __init__(self, num_inputs = None, range_low = None, range_high=None,
    variance=0.1, neuron_json=None):
        """
        Constructs an RBF neuron with a center
        that has num_input dimensions
        """
        if neuron_json is None:
            self.center = np.random.uniform(range_low, range_high, num_inputs)
            self.variance = variance
        else:
            self.center = neuron_json["center"]
            self.variance = neuron_json["variance"]

    def activate(self, inputs):
        """
        There is no activation of the inputs in a RBF neuron.
        """
        pass

    def forward(self, inputs):
        return self.transfer(inputs)

    def transfer(self, inputs):
        """
        A Gaussian function on the inputs.
        """
        distance = np.sqrt(np.sum((self.center - inputs)**2))
        return np.exp(-(distance**2) / (2*(self.variance)**2))
    
    def transfer_derivative(self, inputs):
        """
        We do not need a transfer derivative on this neuron, as
        we do not have any weights.
        """
        pass

    def json(self):
        return dict(
            center=[float(val) for val in self.center],
            variance=self.variance
        )

class RBFNetwork(ArtificialNeuralNetwork):

    def __init__(self, num_inputs, i_lower_bound=-1.5, i_upper_bound=1.5, num_hidden_units = 10,
                 variance=0.5, learning_rate = 0.1):
        self.layers = []
        hidden_layer = [RBFNeuron(num_inputs, i_lower_bound, i_upper_bound, variance) for i in range(num_hidden_units)]
        output_layer = [Neuron(num_hidden_units, transfer_function="linear")]
        self.layers.append(hidden_layer)
        self.layers.append(output_layer)
        self.learning_rate = learning_rate

    def backprop_error(self, expected_outputs):
        """
        Calculates the error on the output layer
        :param expected_outputs:
        :return:
        """
        layer = self.layers[-1]
        for j in range(len(layer)):  # for each neuron in the output layer
            neuron = layer[j]
            error = expected_outputs[j] - neuron.output
            neuron.delta = error * neuron.transfer_derivative(neuron.output)
        return [neuron.delta for neuron in layer]

    def calculate_weight_gradients(self):
        """
        Calculates the gradients of each of the weights
        :return:
        """
        weight_gradients = []
        output_layer = self.layers[1]
        hidden_layer = self.layers[0]
        inputs = [neuron.output for neuron in hidden_layer]  # the inputs are just the inputs from the hidden layer
        for neuron in output_layer:
            neuron_weight_gradients = []
            for j in range(len(inputs)):
                # update the jth weight for the jth input
                neuron_weight_gradients.append(self.learning_rate * neuron.delta * inputs[j])
            weight_gradients.append(neuron_weight_gradients)
        return weight_gradients

    def apply_weight_gradients(self, weight_changes):
        """
        Updates the weights in the output layer by the given weight change vector
        :param weight_changes: An array where each item in the array is a vector
        of weight changes for the neuron in the respective index that the weight change
        vector is in
        """
        output_layer = self.layers[1]
        for i in range(len(weight_changes)): # for each neuron in the output layer
            neuron = output_layer[i]
            dweights = weight_changes[i]
            for j in range(len(neuron.weights)):  # for each weight in the neuron
                neuron.weights[j] += dweights[j]

    def update_weights(self):
        """
        Updates the weights in the output layer by the given weight change
        vector
        """
        changes = self.calculate_weight_gradients()
        self.apply_weight_gradients(changes)

    def train(self, inputs, expected_outputs):
        output = self.forward(inputs)
        self.backprop_error(expected_outputs)
        self.update_weights()
        return output

    def train_without_update(self, inputs, expected_outputs):
        """
        Performs the training process without actually updating the weights of the
        network. Instead, the weight changes will be returned as an array of the
        respective weight changes for each weight in the output layer.
        :param inputs: The data row
        :param expected_outputs: The expected outputs of the data row
        :return: A tuple of the outputs and the weight changes calculated
            from the inputs
        """
        outputs = self.forward(inputs)
        self.backprop_error(expected_outputs)
        dweights = self.calculate_weight_gradients()
        return [outputs, dweights]

class PretrainedRBFNetwork(RBFNetwork):

    def __init__(self, network_json_str):
        network_json = json.loads(network_json_str)
        output_layer = network_json["output"]
        hidden_layers = network_json["hidden"]

        # construct our layers from the
        self.layers = [[RBFNeuron(neuron_json=neuron) for neuron in layer] for layer in hidden_layers]
        self.layers.append([Neuron(neuron_json=neuron) for neuron in output_layer])

class RBFTrainer:

    def __init__(self, network: RBFNetwork, training_set, validation_set, learning_rate=0.1):
        network.set_learning_rate(learning_rate)
        self.network = network
        self.training_set = training_set
        self.validation_set = validation_set
        self.should_train = True

    def train_regression_incremental(self, dataset, learning_rate=0.1, num_epochs=100, start_epoch=0):
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

    def train_regression_batch(self, batch_size=None, learning_rate=0.1, num_epochs=100, start_epoch=0):
        """
        Performs batch learning on the dataset and the given network. Mini-batch can be performed by setting
        the batch size.
        :param dataset: The data points to learn from
        :param batch_size: The size of the batches. If it is not supplied,
            the batch size will just be the size of the dataset
        :param learning_rate: The learning rate
        :param num_epochs: The number of epochs to perform
        :param start_epoch: The starting epoch number
        :return: A generator that yields the epoch and mean squared error of
            all of the training points in dataset during the epoch
        """
        self.network.set_learning_rate(learning_rate)

        batches = None
        if batch_size is None:
            batches = [self.training_set]
        else:
            batches = util.chunk_array(self.training_set, batch_size)

        if num_epochs == -1:
            num_epochs = 1000000000000

        for epoch in range(start_epoch, start_epoch + num_epochs):
            sum_error = 0
            for batch in batches:
                batch_dweights = []  # array of weight changes for data points in this batch
                for row in batch:
                    inputs = row[:-1]
                    expected = row[-1:]
                    [output, dweights] = self.network.train_without_update(inputs, expected) # calculate the outputs
                    sum_error += (expected[0] - output[0]) ** 2
                    self.network.backprop_error(expected)
                    batch_dweights.append(dweights)
                mean_dweights = np.mean(batch_dweights, axis=0) # calculate the mean weight change for each weight
                self.network.apply_weight_gradients(mean_dweights)

            mse_training = sum_error / 2
            mse_validation = util.mean_squared_error(self.validation_set, self.network)
            yield [epoch, mse_training, mse_validation]