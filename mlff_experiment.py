import csv
import sys

from dataset import Datasets
from nn.mlff import *
from util import QueuedCsvWriter, KFoldCrossValidation, mean_squared_error

num_epochs = int(sys.argv[1])
dataset = Datasets.random_rosenbrock(2, points=1000)
network = MLFFNetwork(1, 1, 10, output_transfer="linear")
trainer = MLFFNetworkTrainer(network)
k = KFoldCrossValidation(dataset, 5)
training_set = k.get_training_set(0)
validation_set = k.get_validation_set(0)

writer = QueuedCsvWriter("results.csv", ["epoch", "err"])

def run():
    for [epoch, sum_error] in trainer.train_regression_stochastic(training_set, num_epochs):
        # writer.write_row([str(epoch), "%.3f" % sum_error])
        if epoch % 10 == 0:
            mse = mean_squared_error(validation_set, network)
            print("epoch %d, validation_mse=%.3f, sum_error=%.3f" % (epoch, mse, sum_error))

run()