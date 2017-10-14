import random

class KFoldCrossValidation:

    def __init__(self, dataset, num_folds):
        self.num_folds = num_folds
        random.shuffle(dataset)
        chunk_size = int(len(dataset) / num_folds)
        self.folds = [dataset[i:i+chunk_size] for i in range(0, len(dataset), chunk_size)]

    def get_training_set(self, fold_index):
        result = []
        for i in range(self.num_folds):
            if i == fold_index:
                continue
            result += self.folds[i]
        return result

    def get_validation_set(self, fold_index):
        return self.folds[fold_index]