import argparse
import signal
import sys
from dataset import Datasets

from nn.mlff import *
from experiment import *

parser = argparse.ArgumentParser()
parser.add_argument("num_inputs", help="The number of inputs for the Rosenbrock function to use", type=int)
parser.add_argument("num_points", help="The number of Rosenbrock data points to generate", type=int)
parser.add_argument("num_hidden_layers", help="The number of hidden layers for the network", type=int)
parser.add_argument("num_nodes_layer", help="The number of nodes per hidden layer", type=int)
parser.add_argument("results", help="The results file to save to")
parser.add_argument("models", help="The models file to save to")

if __name__ == "__main__":
    args = parser.parse_args()

    dataset = Datasets.random_rosenbrock(args.num_inputs, args.num_points)
    network = MLFFNetwork(args.num_inputs, args.num_hidden_layers, args.num_nodes_layer)
    experiment = Experiment(network, dataset, args.results, args.models)

    def on_exit(*args):
        print("=== Exit signal received, stopping training prematurely ===")
        experiment.exit_handler()
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_exit)

    try:
        experiment.run()
    except:
        experiment.exit_handler()