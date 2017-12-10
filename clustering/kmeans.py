import random
from collections import deque

import numpy as np

from sklearn.preprocessing import normalize
from clustering import ClusteringAlgorithm
from dataset import Dataset, DatasetLoader
from sklearn import metrics

def euclidean(v1, v2):
    return np.sqrt(sum([(a-b)**2 for a,b in zip(v1, v2)]))

class KMeans(ClusteringAlgorithm):

    def __init__(self,
                 num_inputs,
                 num_outputs,
                 X,
                 Y):
        self.centers = [np.random.random(num_inputs) for i in range(num_outputs)]
        self.X = normalize(X)
        self.Y = Y
        self.dataset_size = len(self.X)

    def predict_single(self, input_pattern):
        center_distances = []
        for index, center in enumerate(self.centers):
            # append the output of the neuron and the neuron weights
            center_distances.append([euclidean(center, input_pattern), index])

        # find max by output and save the neuron's index
        min_index = min(center_distances, key=lambda item: item[0])[1]
        return min_index

    def predict(self, input_patterns):
        """
        Predicts the cluster for each input pattern
        in input_patterns
        :param input_patterns: An array of input patterns
        """
        return [self.predict_single(input_pattern) for input_pattern in input_patterns]

    def run(self, max_epochs=1000):

        for epoch in range(1, max_epochs):
            assignments = [[] for center in self.centers]
            # assign each input pattern to a cluster
            for input_pattern in self.X:
                cluster = self.predict_single(input_pattern)
                assignments[cluster].append(input_pattern)

            has_diff = False    # whether or not there was a change in the clusters

            # update the centers
            for i in range(len(self.centers)):
                if len(assignments[i]) > 0: # only update if cluster has assignments
                    new_center = np.mean(assignments[i], axis=0)
                    if any(np.not_equal(new_center, self.centers[i])):
                        has_diff = True
                        self.centers[i] = new_center

            # print statistics every 10 iterations through the dataset
            if epoch % 10 == 0:
                new_predictions = self.predict(self.X)

                homogeneity = metrics.homogeneity_score(self.Y, new_predictions)
                completeness = metrics.completeness_score(self.Y, new_predictions)
                print("completeness=%f, homogeneity=%f" % (completeness, homogeneity))

                if not has_diff and epoch > 100:
                    print("No change in the clusters, stopping.")
                    break

            epoch += 1


ds = DatasetLoader.load("seeds")
X, Y = ds.X, ds.CLASS_Y
network = KMeans(ds.num_inputs, ds.num_outputs, X, Y)
network.run(100000)