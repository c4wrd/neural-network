import csv
import sys

from dataset import Datasets
from nn.rbf import *
from util import KFoldCrossValidation, CachedWriter

num_epochs = int(sys.argv[1])
dataset = Datasets.random_rosenbrock(2, 500, -1.5, 1.5)
#dataset = Datasets.linear()[0:100]
network = RBFNetwork(2, -1.5, 1.5, 20, 1)
trainer = RBFTrainer(network)
k = KFoldCrossValidation(dataset, 5)
training_set = k.get_training_set(0)
validation_set = k.get_validation_set(0)

results_file = open("results.csv", "w+")
writer = csv.writer(results_file)
writer.writerow(["epoch", "mse", "sum_error"])
writer = CachedWriter(writer)

def run():
    for [epoch, sum_error] in trainer.train_regression(training_set, learning_rate=0.1, num_epochs=num_epochs):
        mse = trainer.mean_squared_error(validation_set)
        writer.write_row([str(epoch), "%f" % mse, "%f" % sum_error])
        # if epoch % 10 == 0:
        #     mse = trainer.mean_squared_error(validation_set)
        #     print("epoch %d, validation_mse=%.3f, sum_error=%.3f" % (epoch, mse, sum_error))
        print("epoch %d, mse=%.3f, sum_error=%.3f" % (epoch, mse, sum_error))

try:
    run()
except:
    writer.flush()
writer.flush()