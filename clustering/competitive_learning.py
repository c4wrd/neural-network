import random

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
                 learning_rate = 0.1):
        self.learning_rate = learning_rate
        self.neuron_weights = np.random.uniform(size=(num_outputs, num_inputs))
        self.X = X
        self.NORM_X = normalize(X)
        self.Y = Y
        self.predictions = [self.predict(input_pattern) for input_pattern in self.NORM_X]

    def predict(self, input_pattern):
        neuron_outputs = []
        for index, weights in enumerate(self.neuron_weights):
            # append the output of the neuron and the neuron weights
            neuron_outputs.append([weights.dot(input_pattern), index])

        # find max by output and save the neuron's index
        max_index = max(neuron_outputs, key=lambda item: item[0])[1]
        return max_index

    def run(self, max_epochs=1000):
        epoch = 0
        while True:
            input_pattern = random.choice(self.NORM_X)
            max_index = self.predict(input_pattern)
            max_weights = self.neuron_weights[max_index]
            weight_update = self.learning_rate*(input_pattern)

            max_weights += weight_update

            # normalize the weights back to unit length
            magnitude = np.sqrt(max_weights.dot(max_weights))
            max_weights /= magnitude

            # print statistics every n iterations
            if epoch % len(self.X) == 0:
                new_predictions = [self.predict(input_pattern) for input_pattern in self.NORM_X]
                homogeneity = metrics.homogeneity_score(self.Y, new_predictions)
                completeness = metrics.completeness_score(self.Y, new_predictions)
                print("completeness=%f, homogeneity=%f" % (completeness, homogeneity))

                # check if no changes have occurred in the clusters
                #if new_predictions == self.predictions:
                #    print("No changes in clusters detected, training completed.")
                #    break

                self.predictions = new_predictions

            epoch += 1


ds = DatasetLoader.load("glass")
X, Y = ds.X, ds.CLASS_Y
network = CompetitiveLearningNeuralNetwork(ds.num_inputs, ds.num_outputs, X, Y, 0.01)
network.run()