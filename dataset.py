import random
from rosenbrock import rosenbrock

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
        data = []
        for i in range(1000):
            val = random.random() * 5
            data.append([val, 2*val + 5])
        return data
            
    @staticmethod
    def random_rosenbrock(numx = 2, points = 100, lower_bound = -3, upper_bound = 3):
        data = []
        adjusted_upper = upper_bound * 100
        adjusted_lower = lower_bound * 100
        for i in range(points):
            values = []
            for j in range(numx):
                values.append(float(random.randint(adjusted_lower, adjusted_upper) / 100))
            result = rosenbrock(*values)
            values.append(result)
            data.append(values)
        return data