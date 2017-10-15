import random
import numpy as np

from nn.transfer import DERIVATIVE, FUNCTION, TRANSFER_FUNCTIONS


def get_average_delta_weight(deltas):
    return []

def get_average_delta(deltas):
    return []

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
        """
        Forwards the inputs, where inputs is a row in a dataset,
        through the neurons in each layer and returns the output layer
        outputs.
        """
        for layer in self.layers:
            outputs = []
            for neuron in layer:
                output = neuron.forward(inputs)
                neuron.output = output
                outputs.append(output)
            inputs = outputs # set the inputs to the outputs of the layer
        return inputs

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

        :param expected_outputs The target outputs of the network
            for which error will be calculated for
        :returns deltas The array of computed deltas when propagating
            the expected_outputs
        """
        self.backprop_output_layer(expected_outputs)
        self.backprop_hidden_layers()
        return [[neuron.delta for neuron in layer] for layer in self.layers]

    def get_delta_weights(self, row, learning_rate):
        """
        Calculates the delta weight change for each weight of each neuron
            in each layer of the network.

        :param row A particular data point for which the weights will
            be adjusted for
        :param learning_rate The learning rate to apply during weight update
            calculations
        """
        delta_weights = []
        for i in range(len(self.layers)):
            layer_weights = []
            layer = self.layers[i]
            inputs = row
            if i != 0:
                prev_layer = self.layers[i-1]
                inputs = [neuron.output for neuron in prev_layer]
            for neuron in layer:
                delta_neuron_weights = []
                for j in range(len(inputs)):
                    # update the jth weight for the jth input
                    delta_neuron_weights.append(learning_rate * neuron.delta * inputs[j])
                layer_weights.append(delta_neuron_weights)
            delta_weights.append(layer_weights)
        return delta_weights

    def update_weights_with_deltas(self, delta_err, delta_weights, learning_rate):
        """
        Updates the weights for each neuron in each layer according
        to the provided delta_err and delta_weight inputs. This is useful
        for batch (or mini batch) gradient descent.
        """
        for l in range(len(delta_weights)):
            layer_weights = delta_weights[l]
            for n in range(len(layer_weights)):
                neuron_weights = layer_weights[n]
                neuron = self.layers[l][n]
                for w in range(len(neuron.weights)):
                    neuron.weights[w] += neuron_weights[w]
                neuron.bias += learning_rate * delta_err[l][n]


    def update_weights(self, row, learning_rate):
        """
        Updates the weights for all of the layers in the network
            from a single data point (stochastic gradient descent).

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

    def train(self, data, expected_outputs, learning_rate=0.1):
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

    def train_batch(self, data, expected_outputs, learning_rate = 0.1):
        # TODO
        batch_deltas = []
        batch_weights = []
        for i in range(len(data)):
            row = data[i]
            expected = expected_outputs[i]
            output = self.forward(row)
            d_errs = self.backprop_error(expected)
            batch_deltas.append(d_errs)
            dweights = self.get_delta_weights(row, learning_rate)
            batch_weights.append(dweights)
        # compute average delta error
        avg_derr = get_average_delta(batch_deltas)
        # compute average delta weight change
        avg_dweights = get_average_delta_weight(batch_weights)
        self.update_weights_with_deltas(avg_derr, avg_dweights, learning_rate)

    def json(self):
        json_network = {
            "output": [neuron.json() for neuron in self.layers[-1]],
            "hidden": []
        }

        if len(self.layers) > 1:
            json_network["hidden"] = [[neuron.json() for neuron in layer] for layer in self.layers[:-1]]
        
        return json_network