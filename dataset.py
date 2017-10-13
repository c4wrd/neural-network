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
            data.append([i, i**2])
        return data