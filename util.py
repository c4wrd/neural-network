import random
from collections import deque

class CachedWriter:
    def __init__(self, writer):
        self.writer = writer
        self.queue = deque()

    def write_row(self, row):
        self.queue.append(row)
        if len(self.queue) == 100:
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