import numpy

from sklearn.metrics import accuracy_score
from nn.mlff import MLFFNetwork

class EvolutionaryStrategy:

    def run_generation(self) -> (int, int):
        raise NotImplementedError()

    def get_fittest_individual(self) -> MLFFNetwork:
        raise NotImplementedError()

def squared_error(data, individual: MLFFNetwork):
    """
    Fitness function to calculate the sum error for
    an individual
    """
    sum_error = 0
    for value in data:
        inputs = value[0]
        expected = value[1]
        outputs = individual.forward(inputs)
        for i in range(len(outputs)):
            sum_error += (expected[i] - outputs[i]) ** 2
    return -(sum_error / 2)

def classification_accuracy(data, individual: MLFFNetwork):
    """
    Fitness function to determine the classification
    accuracy of an individual
    """
    dx, dy = zip(*[(row[0], numpy.argmax(row[1])) for row in data])
    predicted_y = individual.predict(dy, True)
    return accuracy_score(dy, predicted_y)


class FitnessFunctions:

    squared_error = squared_error
    classification_accuracy = classification_accuracy