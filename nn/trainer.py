import util
import numpy as np

from nn.mlff import MLFFNetwork
from nn.neural_network import ArtificialNeuralNetwork
from nn.rbf import RBFNetwork


def get_mean_gradients_mlff(dweights):
    """
    Calculates the mean weight gradient for a supplied
    list of weight gradients acquired over multiple training
    updates. Because the weight gradient is of multiple dimensions,
    we must iterate over each layer, neuron and weight individually to
    calculate the average.
    :param dweights: An array of gradient weight matrices
    :returns The average weight gradients
    """
   # summed_dweights = np.sum(dweights, axis=0)
    summed_dweights = dweights.pop()
    num_dweights = len(dweights)

    # accumulate the weights
    for dweight in dweights:
        for layer_index in range(len(summed_dweights)):
            layer_dweights = summed_dweights[layer_index]
            for neuron_index in range(len(layer_dweights)):
                neuron_dweights = layer_dweights[neuron_index]
                for dweight_index in range(len(neuron_dweights)):
                    weight_gradient = dweight[layer_index][neuron_index][dweight_index]
                    summed_dweights[layer_index][neuron_index][dweight_index] += weight_gradient

    # list comprehension that simply divides each weight in the summed weights to get the mean
    mean_weights = [[[weight / (num_dweights + 1) for weight in neuron] for neuron in layer] for layer in summed_dweights]
    return mean_weights

class NetworkTrainer:

    def __init__(self, network: ArtificialNeuralNetwork, training_set, validation_set, learning_rate=0.1, classification=False, num_classes=None):
        network.set_learning_rate(learning_rate)
        self.network = network
        if not classification:
            self.training_set = training_set
            self.validation_set = validation_set
        else:
            self.training_set = self.transform(training_set, num_classes)
            self.validation_set = self.transform(validation_set, num_classes)
        self.should_train = True

    def train_incremental(self, max_epochs=-1, start_epoch=0, learning_rate = 0.1):
        """
        Trains the network on incrementally.
        """
        self.network.set_learning_rate(learning_rate)
        if max_epochs == -1:
            max_epochs = 1000000000000
        for epoch in range(start_epoch, start_epoch + max_epochs):
            sum_error = 0
            for row in self.training_set:
                inputs = row[0]
                expected = row[1]
                if not isinstance(expected, list):
                    expected = [expected]
                outputs = self.network.train(inputs, expected)
                for i in range(len(outputs)):
                    sum_error += (expected[i] - outputs[i]) ** 2

            mse_training_set = sum_error / 2
            mse_validation_set = util.mean_squared_error(self.validation_set, self.network)

            # yield the epoch and errors to determine whether training should continue
            yield [epoch, mse_training_set, mse_validation_set]

    def train_batch(self, learning_rate=0.1, max_epochs=-1, start_epoch=0, batch_size=None):
        self.network.set_learning_rate(learning_rate)

        batches = None
        if batch_size is None:
            batches = [self.training_set]
        else:
            batches = util.chunk_array(self.training_set, batch_size)

        if max_epochs == -1:
            max_epochs = 1000000000000

        for epoch in range(start_epoch, start_epoch + max_epochs):

            # determine whether we should continue training
            if not self.should_train:
                break

            sum_error = 0

            # update the network for each batch
            for batch in batches:
                batch_dweights = []  # array of weight changes for data points in this batch
                for row in batch:
                    inputs = row[0]
                    expected = row[1]
                    if not isinstance(expected, list):
                        expected = [expected]
                    [outputs, dweights] = self.network.train_without_update(inputs, expected)  # calculate the outputs
                    batch_dweights.append(dweights)
                    for i in range(len(outputs)):
                        sum_error += (expected[i] - outputs[i]) ** 2

                # compute the mean weight gradient matrix
                mean_dweights = self.get_mean_gradients(batch_dweights)
                self.network.apply_weight_gradients(mean_dweights)

            mse_test_set = sum_error / 2    # mean squared error of the training set
            mse_validation_set = util.mean_squared_error(self.validation_set, self.network) # mse of validation set

            # yield the epoch, mean square errors to determine whether training should continue
            yield [epoch, mse_test_set, mse_validation_set]

    def get_mean_gradients(self, gradients):
        """
        Computes the mean gradients given an array of weight
        update matrices

        :param gradients: An array of weight update matrices
        :return: The mean weight update matrix
        """
        if isinstance(self.network, MLFFNetwork):
            return get_mean_gradients_mlff(gradients)
        else:
            return np.mean(gradients, axis=0)

    def stop(self):
        """
        Stops the current training process, because we determined either the model has reached
        a minimum, or the validation data set error is starting to increase, suggesting the model
        is overfitting.
        """
        self.should_train = False

    def transform(self, dataset, num_classes):
        fixed = []
        for row in dataset:
            features = row[:-1]
            expected_class = row[-1]
            expected = [0 for i in range(num_classes)]
            expected[expected_class] = 1
            fixed.append([*features, expected])
        return fixed