import random
import csv
from collections import deque

from nn.neural_network import ArtificialNeuralNetwork


class QueuedCsvWriter:
    """
    Wrapper around the built-in CSV writer
    that will write lines in blocks for performance
    reasons.
    """
    def __init__(self, result_file_name, header_row, queue_size = 50):
        self.file = open(result_file_name, "w+")
        self.writer = csv.writer(self.file)
        self.writer.writerow(header_row)
        self.queue = deque()
        self.queue_size = queue_size

    def writerow(self, row):
        self.queue.append(row)
        if len(self.queue) >= self.queue_size:
            while self.queue:
                self.writer.writerow(self.queue.popleft())

    def flush(self):
        while self.queue:
            self.writer.writerow(self.queue.popleft())


class KFoldCrossValidation:

    def __init__(self, dataset, num_folds):
        """
        Constructs a KFoldCrossValidation strategy.

        :param dataset The dataset to construct folds for
        :param num_folds The number of folds to use
        """
        self.num_folds = num_folds
        random.shuffle(dataset) # create random samples
        chunk_size = int(len(dataset) / num_folds)
        # create chunks where each chunk is a fold of chunk_size
        self.folds = [dataset[i:i+chunk_size] for i in range(0, len(dataset), chunk_size)]

    def get_training_set(self, fold_index):
        """
        Retreives the training set from the data set, where
        each fold is returned besides the specified fold_index

        :fold_index The fold to withhold
        """
        result = []
        for i in range(self.num_folds):
            if i == fold_index:
                continue
            result += self.folds[i]
        return result

    def get_validation_set(self, fold_index):
        """
        Returns the validation data set, where the
        validation data set is the fold_index index
        in the dataset.

        :param fold_index The fold to retrieve
        """
        return self.folds[fold_index]

def mean_squared_error(dataset, network: ArtificialNeuralNetwork):
    sum_error = 0
    for row in dataset:
        inputs = row[:-1]
        expected = row[-1]
        if not isinstance(expected, list):
            expected = [expected]
        outputs = network.forward(inputs)
        for i in range(len(outputs)):
            sum_error += (expected[i] - outputs[i])**2
    return sum_error / 2

def chunk_array(array, chunk_size):
    """
    Splits an array into chunk_size subarrays
    :param array: The array to split
    :param chunk_size: The max size of each chunk_array
    :return:
    """
    chunks = []
    for i in range(0, len(array), chunk_size):
        chunks.append(array[i:i+chunk_size])
    return chunks