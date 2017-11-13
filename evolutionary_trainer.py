import argparse
import signal
import sys

from dataset import Datasets, DatasetLoader, DatasetType
from nn.diff_ev import DETrainer
from nn.ga import GATrainer
from experiment import *
from nn.mpl import MPLTrainer
from nn.neural_network import NetworkShape

parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", help="The dataset to load", type=str)
parser.add_argument("strategy", help="Which training strategy to use [mpl, ga, de]", type=str)
parser.add_argument("num_hidden_layers", help="The number of hidden layers for the network", type=int)
parser.add_argument("num_nodes_layer", help="The number of nodes per hidden layer", type=int)
parser.add_argument("results", help="The results file to save to")
parser.add_argument("models", help="The models file to save to")
parser.add_argument("-p", "--population_size", help="The population size for GA or mu parameter for ES", type=int)
parser.add_argument("-t", "--tournament_size", help="The tournament size for GA", type=int)
parser.add_argument("-a", "--alpha", help="The probability of crossover for DE", type=float)
parser.add_argument("-b", "--beta", help="The beta parameter for DE", type=float)
parser.add_argument("-l", "--lambda_parameter", help="The lambda parameter for ES", type=int)


if __name__ == "__main__":
    args = parser.parse_args()

    dataset = DatasetLoader.load(args.dataset_name)
    output_transfer = "linear" if dataset.type == DatasetType.REGRESSION else "logistic"
    shape = NetworkShape(dataset.num_inputs, args.num_hidden_layers, args.num_nodes_layer, dataset.num_outputs)
    if args.strategy == "mpl":
        if args.population_size is None:
            raise Exception("Population size must be specified (mu)")
        if args.lambda_parameter is None:
            raise Exception("Lambda parameter must be specified")
        trainer = MPLTrainer(shape=shape,
                             mu_size=args.population_size,
                             lambda_size=args.lambda_parameter,
                             dataset=dataset,
                             output_transfer=output_transfer,
                             tournament_size=args.tournament_size)
    elif args.strategy == "ga":
        if args.population_size is None:
            raise Exception("Population size must be specified")
        trainer = GATrainer(network_shape=shape,
                            pop_size=args.population_size,
                            dataset=dataset,
                            output_transfer=output_transfer,
                            tournament_size=args.tournament_size)
    elif args.strategy == "de":
        if args.population_size is None:
            raise Exception("Population size must be specified")
        if args.alpha is None:
            raise Exception("Alpha parameter (crossover rate) must be specified")
        if args.beta is None:
            raise Exception("Beta parameter must be specified")
        trainer = DETrainer(
            ps=args.population_size,
            pop_s=shape,
            alpha=args.alpha,
            beta=args.beta,
            dataset=dataset
        )
    else:
        raise Exception("%s is not a valid training strategy" % args.strategy)

    experiment = EvolutionaryExperiment(trainer, dataset, args.results, args.models)

    def on_exit(*args):
        print("=== Exit signal received, stopping training prematurely ===")
        experiment.exit_handler()
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_exit)

    try:
        experiment.run()
    except:
        experiment.exit_handler()