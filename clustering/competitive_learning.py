import random

import numpy as np

from sklearn.preprocessing import normalize
from clustering import ClusteringAlgorithm
from dataset import Dataset, DatasetLoader
from sklearn import metrics

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
from matplotlib import pyplot as plt
plt.ion()
plt.show()

class CompetitiveLearningNeuralNetwork(ClusteringAlgorithm):

    def __init__(self,
                 num_inputs,
                 num_outputs,
                 X,
                 Y,
                 learning_rate = 0.1):
        self.learning_rate = learning_rate
        self.neuron_weights = [np.random.rand(num_inputs) for i in range(num_outputs)]

        self.X = normalize(X)
        self.Y = Y

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
            input_pattern = self.X[0] #random.choice(self.X)
            max_index = self.predict(input_pattern)
            max_weights = self.neuron_weights[max_index]
            weight_update = self.learning_rate*(input_pattern)

            max_weights += weight_update

            # normalize the weights back to unit length
            magnitude = np.sqrt(max_weights.dot(max_weights))
            max_weights /= magnitude

            # print mean weight update at end of each epoch
            if epoch % 10000 == 0:
                self.print_stats()

            epoch += 1

    def print_stats(self):
        predictions = [self.predict(input_pattern) for input_pattern in self.X]
        homogeneity = metrics.homogeneity_score(self.Y, predictions)
        completeness = metrics.completeness_score(self.Y, predictions)
        print("completeness=%f, homogeneity=%f" % (completeness, homogeneity))
        Xs = [x[0] for x in self.X]
        Ys = [x[1] for x in self.X]
        cls = [colors[i] for i in predictions]
        plt.scatter(Xs, Ys, color=cls)
        plt.draw()
        plt.pause(0.001)


from sklearn import datasets

X, Y = datasets.make_blobs(n_samples=1000, centers=len(colors), cluster_std=0.2)
network = CompetitiveLearningNeuralNetwork(2, len(colors), X, Y, 0.01)
network.run()