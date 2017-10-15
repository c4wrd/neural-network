import csv
import sys

from dataset import Datasets
from nn.rbf import *
from util import KFoldCrossValidation, QueuedCsvWriter, mean_squared_error

num_epochs = int(sys.argv[1])
dataset = Datasets.random_rosenbrock(2, 500, -1.5, 1.5)
network = RBFNetwork(2, -1.5, 1.5, 50, 0.5)
trainer = RBFTrainer(network)
k = KFoldCrossValidation(dataset, 5)
training_set = k.get_training_set(0)
validation_set = k.get_validation_set(0)
writer = QueuedCsvWriter("results.csv", ["epoch, mse_valid, mse_train"])

def run():
    for [epoch, sum_error] in trainer.train_regression_batch(training_set, batch_size=100, learning_rate=0.1, num_epochs=num_epochs):
        mse = mean_squared_error(validation_set, network)
        print("epoch %d, mse=%.3f, sum_error=%.3f" % (epoch, mse, sum_error))
        writer.writerow([str(epoch), "%f" % mse, "%f" % sum_error])
            # if epoch % 10 == 0:
            #     mse = trainer.mean_squared_error(validation_set)
            #     print("epoch %d, validation_mse=%.3f, sum_error=%.3f" % (epoch, mse, sum_error))

run()