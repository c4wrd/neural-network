import csv
import sys

from dataset import Datasets
from nn.mlp import *
from util import CachedWriter, KFoldCrossValidation

num_epochs = int(sys.argv[1])
dataset = Datasets.random_rosenbrock(2, points=1000)
dataset = Datasets.linear()
network = MLFFNetwork(1, 1, 10, output_transfer="linear")
trainer = MLPNetworkTrainer(network)
k = KFoldCrossValidation(dataset, 5)
training_set = k.get_training_set(0)
validation_set = k.get_validation_set(0)

results_file = open("results.csv", "w+")
writer = csv.writer(results_file)
writer.writerow(["epoch", "mse"])
writer = CachedWriter(writer)

def run():
    for [epoch, sum_error] in trainer.train_linear_regression(training_set, num_epochs):
        # writer.write_row([str(epoch), "%.3f" % sum_error])
        if epoch % 10 == 0:
            mse = trainer.mean_squared_error(validation_set)
            print("epoch %d, validation_mse=%.3f, sum_error=%.3f" % (epoch, mse, sum_error))

try:
    run()
except:
    writer.flush()
writer.flush()