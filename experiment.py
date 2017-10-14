from mlp import *
from neural_network import *
from dataset import Datasets
from kfold import *
import sys
import csv

num_epochs = int(sys.argv[1])
dataset = Datasets.random_rosenbrock(2, points=1000)
network = MLFFNetwork(2, 1, 15, output_transfer="linear")
trainer = MLPNetworkTrainer(network)
k = KFoldCrossValidation(dataset, 5)
training_set = k.get_training_set(0)
validation_set = k.get_validation_set(0)

results_file = open("results.csv", "w+")
writer = csv.writer(results_file)
writer.writerow(["epoch", "mse", "validation_mse"])

prev_valid_err = None
for [epoch, sum_error] in trainer.train_linear_regression(training_set, num_epochs):
    mse = trainer.mean_squared_error(validation_set)
    writer.writerow([str(epoch), str(sum_error), str(mse)])
    if epoch % 10 == 0:
        print("epoch %d, sum_error=%.3f" % (epoch, sum_error))
        if prev_valid_err is None:
            prev_valid_err = mse
            print("validation mse: %.3f" % mse)
        else:
            print("validation mse: %.3f, difference: %.3f" % (mse, prev_valid_err - mse))
            prev_valid_err = mse