import random
from collections import deque

import numpy as np

from sklearn.preprocessing import normalize
from clustering import ClusteringAlgorithm
from dataset import Dataset, DatasetLoader
from sklearn import metrics

class CompetitiveLearningNeuralNetwork(ClusteringAlgorithm):

    def __init__(self,
                 num_inputs,
                 num_outputs,
                 X,
                 Y,
                 learning_rate = 0.1,
                 losing_factor = 0.1):
        self.learning_rate = learning_rate
        self.losing_factor = losing_factor
        self.neuron_weights = np.random.random(size=(num_outputs, num_inputs))
        self.X = normalize(X)
        self.Y = Y
        self.dataset_size = len(self.X)

    def predict_single(self, input_pattern):
        neuron_outputs = []
        for index, weights in enumerate(self.neuron_weights):
            # append the output of the neuron and the neuron weights
            neuron_outputs.append([weights.dot(input_pattern), index])

        # find max by output and save the neuron's index
        max_index = max(neuron_outputs, key=lambda item: item[0])[1]
        return max_index

    def predict(self, input_patterns):
        return [self.predict_single(input_pattern) for input_pattern in input_patterns]

    def run(self, max_epochs=1000):

        predictions = self.predict(self.X)
        for epoch in range(1, max_epochs):
            # randomly select dataset_size input patterns to train during this epoch
            input_patterns = [random.choice(self.X) for i in range(self.dataset_size)]

            for input_pattern in input_patterns:
                max_index = self.predict_single(input_pattern) # index of the neuron closest to input pattern
                weight_update = self.learning_rate*(input_pattern)
                for i, weights in enumerate(self.neuron_weights):
                    # using leaky learning, we'll prevent dead units
                    if i == max_index:
                        weights += weight_update
                    else:
                        weights += self.losing_factor*weight_update # update losing neurons with a smaller weight update
                    # normalize weights so they're unit length
                    magnitude = np.sqrt(weights.dot(weights))
                    weights /= magnitude

            # print statistics every 10 iterations through the dataset
            if epoch % 10 == 0:
                new_predictions = self.predict(self.X)

                homogeneity = metrics.homogeneity_score(self.Y, new_predictions)
                completeness = metrics.completeness_score(self.Y, new_predictions)
                print("epoch=%d, completeness=%f, homogeneity=%f" % (epoch, completeness, homogeneity))

                # check if no changes have occurred in the cluster assignments
                if new_predictions == predictions and epoch > 10:
                    print("No changes in clusters detected, training completed after %d epochs." % epoch)
                    break

                # update the predictions
                predictions = new_predictions

            epoch += 1


ds = DatasetLoader.load("seeds")
X, Y = ds.X, ds.CLASS_Y
network = CompetitiveLearningNeuralNetwork(ds.num_inputs, ds.num_outputs, X, Y, 0.01)
network.run()