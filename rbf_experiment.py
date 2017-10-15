import sys

from dataset import Datasets
from nn.rbf import *
from nn.trainer import NetworkTrainer
from util import KFoldCrossValidation, QueuedCsvWriter

num_epochs = int(sys.argv[1])
dataset = Datasets.random_rosenbrock(2, 1000, -1.5, 1.5)
network = RBFNetwork(2, -1.5, 1.5, 50, 0.5)
trainer = NetworkTrainer(network, dataset[:400], dataset[400:])
writer = QueuedCsvWriter("results.csv", ["epoch", "mse_valid", "mse_train"])

def run():
    for [epoch, mse_training, mse_validation] in trainer.train_regression_batch():
        print("epoch %d, mse_train=%.3f, mse_validation=%.3f" % (epoch, mse_training, mse_validation))
        #writer.writerow([str(epoch), "%f" % mse_training, "%f" % mse_validation])
            # if epoch % 10 == 0:
            #     mse = trainer.mean_squared_error(validation_set)
            #     print("epoch %d, validation_mse=%.3f, sum_error=%.3f" % (epoch, mse, sum_error))

run()