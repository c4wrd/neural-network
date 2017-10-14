from mlp import *
from neural_network import *
from dataset import Datasets
from kfold import *
from collections import deque
import sys
import csv

class CachedWriter:
    
    def __init__(self, writer):
        self.writer = writer
        self.queue = deque()
    
    def write_row(self, row):
        self.queue.append(row)
        if len(self.queue) == 100:
            while len(self.queue) > 0:
                self.writer.writerow(self.queue.popleft())

    def flush(self):
        while len(self.queue) > 0:
                self.writer.writerow(self.queue.popleft())

num_epochs = int(sys.argv[1])
dataset = Datasets.random_rosenbrock(2, points=1000)
network = MLFFNetwork(2, 1, 10, output_transfer="linear")
trainer = MLPNetworkTrainer(network)
k = KFoldCrossValidation(dataset, 5)
training_set = k.get_training_set(0)
validation_set = k.get_validation_set(0)

results_file = open("results.csv", "w+")
writer = csv.writer(results_file)
writer.writerow(["epoch", "mse", "validation_mse"])
writer = CachedWriter(writer)

def run():
    prev_err = None
    for [epoch, sum_error] in trainer.train_linear_regression(training_set, num_epochs):
        writer.write_row([str(epoch), "%.3f" % sum_error])
        if epoch % 10 == 0:
            if prev_err is None:
                prev_err = sum_error
                print("epoch %d, sum_error=%.3f" % (epoch, sum_error))
            else:
                diff = prev_err - sum_error
                print("epoch %d, sum_error=%.3f, diff: %.3f" % (epoch, sum_error, diff))
                prev_err = sum_error

try:
    run()
except:
    writer.flush()