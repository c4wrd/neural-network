import random
import numpy as np

def rosenbrock(*x):
    x_total = len(x)
    sum = 0
    for i in range(x_total - 1):
        lhs = (1 - x[i])**2
        rhs = 100*(x[i+1] - x[i]**2)**2
        sum += lhs + rhs
    return sum

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
    def squares():
        data = []
        for i in range(1000):
            val = random.random()
            data.append([val, val**2])
        return data

    @staticmethod
    def linear():
        """
        Computes a random dataset composed of a specified
        linear function for testing backpropagation
        """
        data = []
        for i in range(1000):
            val = random.random() * 5
            data.append([val, 2*val + 5])
        return data
            
    @staticmethod
    def random_rosenbrock(numx = 2, points = 100, lower_bound = -3, upper_bound = 3,
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
            data.append(np.append(values, result))
        return data