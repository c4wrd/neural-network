import random

from functools import partial

from multiprocessing import Pool

from dataset import Dataset
from nn.evolution import EvolutionaryStrategy, squared_error, classification_accuracy, FitnessFunctions
from nn.mlff import MLFFNetwork
from nn.neural_network import NetworkShape

"""
Trains a given neural network shape on the given dataset
using a (mu + lambda) evolution strategy.
"""
class MPLTrainer(EvolutionaryStrategy):

    def __init__(self,
                 network_shape: NetworkShape,
                 mu_size: int,
                 lambda_size: int,
                 dataset: Dataset,
                 output_transfer="logistic",
                 hidden_transfer="logistic",
                 crossover_rate: float = 0.4,
                 mutation_rate: float = 0.1,
                 tournament_size: int = 3,
                 fitness_function = FitnessFunctions.squared_error):
        self.population = []
        self.population_fitness = {}
        self.mu_size = mu_size
        self.lambda_size = lambda_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.network_shape = network_shape
        self.output_transfer = output_transfer
        self.hidden_transfer = hidden_transfer
        self.tournament_size = tournament_size
        self.dataset = dataset
        self.train_data = dataset.get_train()
        self.test_data = dataset.get_validation()
        self.pool = Pool()
        self.fitness_function = fitness_function
        self.__init_population__()
    
    def __init_population__(self):
        """
        Initializes our initial mu individuals
        """
        for i in range(self.mu_size):
            network = self.create_individual()
            self.update_fitness(network)
            self.population.append(network)

    def create_individual(self):
        return MLFFNetwork(
            num_inputs=self.network_shape.num_inputs,
            num_hidden_layers=self.network_shape.num_hidden_layers,
            num_nodes_layer=self.network_shape.num_hidden_nodes,
            num_outputs=self.network_shape.num_outputs,
            output_transfer=self.output_transfer,
            hidden_transfer=self.hidden_transfer
        )

    def select_parent(self):
        candidates = [random.choice(self.population) for i in range(self.tournament_size)]
        return max(candidates, key=lambda candidate: self.population_fitness[candidate])

    def update_fitness(self, individual, population_fitness=None):
        """
        Updates the fitness of the individual. If population_fitness is none,
        it updates the population fitness in this self instance.
        :param individual: individual to evaluate
        :param population_fitness: dictionary to store the fitness, if
            computing in parallel
        :return:
        """
        fitness = self.fitness_function(self.train_data, individual)
        if population_fitness is None:
            self.population_fitness[individual] = fitness
        else:
            population_fitness[individual] = fitness
        return fitness

    def get_fittest_individual(self):
        # returns the fittest network
        return max(self.population_fitness.items(), key=lambda pair: pair[1])[0]

    def run_generation(self):

        # generate our child population
        child_population = []
        while len(child_population) <= self.lambda_size:
            p1 = self.select_parent()
            p2 = self.select_parent()
            c1, c2 = self.single_point_crossover(p1, p2)
            self.mutate(c1)
            self.mutate(c2)
            child_population.extend([c1, c2])

        # compute the fitness of each of the children in parallel
        fitness_fn = partial(self.fitness_function, self.train_data)
        fitness_results = self.pool.map(fitness_fn, child_population)
        for individual, fitness in zip(child_population, fitness_results):
            self.population_fitness[individual] = fitness
        
        # sort individuals by their fitness "survival of the fittest"
        all_individuals = sorted(self.population_fitness.items(), key=lambda tuple: tuple[1])

        # store our fittest individual for progress over time
        min_fitness  = all_individuals[-1][1]

        # pop off our mu individuals that are the fittest
        new_population = [all_individuals.pop() for i in range(self.mu_size)]

        # remove old networks from being tracked
        for individual, fitness in all_individuals:
            self.population_fitness.pop(individual)

        # set population to new population
        self.population = [individual for individual, fitness in new_population]

        validation_fitness = self.fitness_function(self.test_data, self.get_fittest_individual())

        return min_fitness, validation_fitness

    def single_point_crossover(self, p1: MLFFNetwork, p2: MLFFNetwork):
        c1 = self.create_individual()
        c2 = self.create_individual()

        # our feature vector is all of the weights and biases
        num_features = p1.num_weights + p1.num_bias_weights

        # choose crossover point
        crossover_point = random.randint(0, num_features)

        # tracks the index of the weight we are currently crossing over
        current_index = 0

        # number of hidden layers + output layer
        num_layers = self.network_shape.num_hidden_layers + 1

        # iterate over each layer
        for p1_layer, p2_layer, layer_index in zip(p1.layers, p2.layers, range(num_layers)):
            num_neurons = len(p1_layer)

            # iterate over each neuron
            for n1, n2, neuron_index in zip(p1_layer, p2_layer, range(num_neurons)): # need to index based
                p1_weights = n1.get_weights()
                p2_weights = n2.get_weights()

                num_weights = len(p1_weights)

                for weight_index in range(num_weights):
                    if current_index == crossover_point:
                        c1.layers[layer_index][neuron_index].set_weight(weight_index, p1_weights[weight_index])
                        c2.layers[layer_index][neuron_index].set_weight(weight_index, p2_weights[weight_index])
                    else:
                        c1.layers[layer_index][neuron_index].set_weight(weight_index, p2_weights[weight_index])
                        c2.layers[layer_index][neuron_index].set_weight(weight_index, p1_weights[weight_index])
                    current_index += 1

        return (c1, c2)

    def arithmetic_crossover(self, p1: MLFFNetwork, p2: MLFFNetwork):
        c1 = self.create_individual()
        c2 = self.create_individual()

        # number of hidden layers + output layer
        num_layers = self.network_shape.num_hidden_layers + 1

        # iterate over each layer
        for p1_layer, p2_layer, layer_index in zip(p1.layers, p2.layers, range(num_layers)):
            num_neurons = len(p1_layer)

            # iterate over each neuron
            for n1, n2, neuron_index in zip(p1_layer, p2_layer, range(num_neurons)): # need to index based
                p1_weights = n1.get_weights()
                p2_weights = n2.get_weights()
                num_weights = len(p1_weights)

                for weight_index in range(num_weights):
                    c1_weight = self.crossover_rate*p1_weights[weight_index] + (1-self.crossover_rate)*p2_weights[weight_index]
                    c2_weight = self.crossover_rate * p2_weights[weight_index] + (1 - self.crossover_rate) * p1_weights[weight_index]

                    c1.layers[layer_index][neuron_index].set_weight(weight_index, c1_weight)
                    c2.layers[layer_index][neuron_index].set_weight(weight_index, c2_weight)

        return (c1, c2)

    def mutate(self, individual: MLFFNetwork):
        """
        Mutates the specified individual, based on the parameters
        specified in the constructor of GATrainer.
        :param individual: The individual to mutate
        :return: None, the individual is mutated in place
        """

        for layer in individual.layers:
            for neuron in layer:
                weights = neuron.get_weights()

                # iterate over the weights
                for wi in range(len(weights)):
                    # determine whether or not we should mutate
                    should_mutate = random.random() <= self.mutation_rate
                    if should_mutate:
                        weight = weights[wi]
                        neuron.set_weight(wi, weight + random.uniform(-0.1, 0.1))