import random
import numpy as np
import csv
import math

def rosenbrock(*x):
    x_total = len(x)
    sum = 0
    for i in range(x_total - 1):
        lhs = (1 - x[i])**2
        rhs = 100*(x[i+1] - x[i]**2)**2
        sum += lhs + rhs
    return sum

class DatasetType:

    CLASSIFICATION = "classification"
    REGRESSION = "regression"

class Dataset:

    def __init__(self, dataset, type, num_inputs, num_outputs):
        self.dataset = dataset
        self.size = len(dataset)
        self.type = type
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.X, self.Y = zip(*[(row[0], row[1]) for row in dataset])
        if type == DatasetType.CLASSIFICATION:
            self.CLASS_Y = [np.argmax(row) for row in self.Y]

    def get_train(self, portion=0.7):
        train_split = math.ceil(portion * self.size)
        return self.dataset[:train_split]

    def get_validation(self, portion=0.3):
        test_split = math.floor((1.0-portion)*self.size)
        return self.dataset[test_split:]

    def get_features(self, point):
        return self.X[point]

    def get_expected_output(self, point):
        return self.Y[point]

    def get_expected_class(self, point):
        return self.CLASS_Y[point]

class Datasets:

    @staticmethod
    def xor():
        return [
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ]

    @staticmethod
    def squares(num_points=1000, low=0, high=1000):
        data = []
        for i in range(num_points):
            val = random.randint(low, high)
            data.append([val, val**2])
        return data

    @staticmethod
    def linear(num_points=1000, low=0, high=5):
        """
        Computes a random dataset composed of a specified
        linear function for testing backpropagation
        """
        data = []
        xvals = np.random.uniform(low, high, num_points)
        for val in xvals:
            data.append([[val], [2*val + 5]])
        return data
            
    @staticmethod
    def random_rosenbrock(numx = 2, points = 100, lower_bound = -2, upper_bound = 2,
        num_decimals=2):
        """
        Computes a random dataset of points size in the range
        of [lower_bound,upper_bound] of the Rosenbrock function
        with numx x values.

        :param numx The number of x parameters
        :param points The number of datapoints to generate
        :param lower_bound The lower bound of the data points
        :param upper_bound The upper bound of the data points
        """
        data = []
        for i in range(points):
            values = np.random.uniform(lower_bound, upper_bound, numx)
            result = rosenbrock(*values)
            data.append((*values, [result]))
        return data

    @staticmethod
    def seeds():
        f = open("dataset_files/seeds.txt")
        writer = csv.reader(f.readlines(), delimiter="\t")
        data = []
        for row in writer:
            try:
                inputs = [float(val) for val in row[:-1] if len(val.strip()) > 0]
                expected_class = int(row[-1]) - 1
                expected_outputs = [0, 0, 0]
                expected_outputs[expected_class] = 1
                data.append([inputs, expected_outputs])
            except:
                print(row)
        random.shuffle(data)
        return Dataset(data, DatasetType.CLASSIFICATION, 7, 3)

class DatasetLoader:

    DATASETS = {
        "seeds": Datasets.seeds
    }

    @classmethod
    def load(self, dataset_name) -> Dataset:
        """
        Returns a specified dataset by name
        """
        if dataset_name not in DatasetLoader.DATASETS:
            raise Exception("Dataset '%s' was not found" % dataset_name)
        return DatasetLoader.DATASETS[dataset_name]()