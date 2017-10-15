import json

from nn.neural_network import ArtificialNeuralNetwork, Neuron


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

    def mean_squared_error(self, dataset):
        sum_error = 0
        for row in dataset:
            inputs = row[:-1]
            expected = row[-1:]
            output = self.network.forward(inputs)
            sum_error += (expected[0] - output[0])**2
        return sum_error / 2

    def train_linear_regression(self, dataset, num_epochs=1000, start_epoch=0, learning_rate = 0.1):
        """
        Train for function approximation (one linear output)
        """
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
            # print(">epoch=%d, lrate=%.3f, error=%.3f" % (epoch, learning_rate, sum_error))

    def train_linear_regression_batch(self, dataset, max_epochs=1000, learning_rate=0.1):
        for epoch in range(max_epochs):
            sum_error = 0
            delta_weights = []
            for row in dataset:
                inputs = row[:-1]
                expected = row[-1]

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
                yield dict(epoch=epoch, error=sum_error)
            # print('>epoch=%d, lrate=%.3f, sum_error=%.3f' % (epoch, learning_rate, sum_error))
