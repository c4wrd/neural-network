import argparse
import signal
import sys

from dataset import DatasetLoader, DatasetType
from nn.mlff import *
from experiment import *

parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", help="The dataset to load", type=str)
parser.add_argument("num_hidden_layers", help="The number of hidden layers for the network", type=int)
parser.add_argument("num_nodes_layer", help="The number of nodes per hidden layer", type=int)
parser.add_argument("results", help="The results file to save to")
parser.add_argument("models", help="The models file to save to")

if __name__ == "__main__":
    args = parser.parse_args()

    dataset = DatasetLoader.load(args.dataset_name)
    output_transfer = "linear" if dataset.type == DatasetType.REGRESSION else "logistic"
    network = MLFFNetwork(dataset.num_inputs,
                          num_hidden_layers=args.num_hidden_layers,
                          num_nodes_layer=args.num_nodes_layer,
                          num_outputs=dataset.num_outputs,
                          output_transfer=output_transfer)
    experiment = BackpropExperiment(network, dataset, args.results, args.models)

    def on_exit(*args):
        print("=== Exit signal received, stopping training prematurely ===")
        experiment.exit_handler()
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_exit)

    try:
        experiment.run()
    except Exception as e:
        experiment.exit_handler()