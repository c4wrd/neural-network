import numpy as np

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
            center=self.center,
            variance=self.variance   
        )

class RBFNetwork(ArtificialNeuralNetwork):

    def __init__(self, num_inputs, i_lower_bound, i_upper_bound, num_hidden_units = 10,
                 variance=0.1, output_transfer="linear"):
        self.layers = []
        hidden_layer = [RBFNeuron(num_inputs, i_lower_bound, i_upper_bound, variance) for i in range(num_hidden_units)]
        output_layer = [Neuron(num_hidden_units, transfer_function="linear")]
        self.layers.append(hidden_layer)
        self.layers.append(output_layer)

    def update_weights_output_layer(self, learning_rate=0.1):
        """
        Performs gradient descent on the output layer
        :param learning_rate:
        :return:
        """
        output_layer = self.layers[1]
        hidden_layer = self.layers[0]
        inputs = [neuron.output for neuron in hidden_layer]
        for neuron in output_layer:
            for j in range(len(inputs)):
                # update the jth weight for the jth input
                neuron.weights[j] += learning_rate * neuron.delta * inputs[j]

    def train(self, inputs, expected_outputs, learning_rate = 0.1):
        output = self.forward(inputs)
        self.backprop_output_layer(expected_outputs)
        self.update_weights_output_layer(learning_rate)
        return output

class RBFTrainer:

    def __init__(self, network):
        self.network = network

    def mean_squared_error(self, dataset):
        sum_error = 0
        for row in dataset:
            inputs = row[:-1]
            expected = row[-1:]
            output = self.network.forward(inputs)
            sum_error += (expected[0] - output[0])**2
        return sum_error / 2

    def train_regression(self, dataset, learning_rate=0.1, num_epochs=100, start_epoch=0):
        if num_epochs == -1:
            num_epochs = 1000000000000
        for epoch in range(start_epoch, start_epoch + num_epochs):
            sum_error = 0
            for row in dataset:
                inputs = row[:-1]
                expected = row[-1:]
                output = self.network.train(inputs, expected, learning_rate)[0]
                sum_error += (expected - output)**2
            yield [epoch, sum_error/2]