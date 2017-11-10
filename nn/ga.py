import random

from nn.mlff import MLFFNetwork

"""
N = population size
P = create parent population by randomly creating N individuals
while not done
    C = create empty child population
    while not enough individuals in C
        parent1 = select parent   ***** HERE IS WHERE YOU DO TOURNAMENT SELECTION *****
        parent2 = select parent   ***** HERE IS WHERE YOU DO TOURNAMENT SELECTION *****
        child1, child2 = crossover(parent1, parent2)
        mutate child1, child2
        evaluate child1, child2 for fitness
        insert child1, child2 into C
    end while
    P = combine P and C somehow to get N new individuals
end while
"""

class GATrainer:

    def __init__(self, 
                network_shape,        
                pop_size,
                data,
                output_transfer="logistic",
                crossover_rate=0.5,
                mutation_rate=0.1,
                tournament_size=3,
                patience=25):
        self.population = []
        self.population_fitness = {}
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.network_shape = network_shape
        self.output_transfer = output_transfer
        self.tournament_size = tournament_size
        self.patience = patience
        self.data = data
        self.__init_population__()
    
    def __init_population__(self):
        for i in range(self.pop_size):
            network = self.create_individual()
            self.population.append(network)

    def create_individual(self):
        return MLFFNetwork(
            num_inputs=self.network_shape[0],
            num_hidden_layers=self.network_shape[1],
            num_nodes_layer=self.network_shape[2],
            num_outputs=self.network_shape[3],
            output_transfer=self.output_transfer
        )


    def select_parent(self):
        candidates = [random.choice(self.population) for i in range(self.tournament_size)]
        return max(candidates, lambda candidate: self.population_fitness[candidate])
    
    def calculate_fitness(self, individual):
        """
        Calculates the fitness as a negative
        mean squared error
        """
        sum_error = 0
        for x in self.data:
            inputs = x[0]
            expected = x[1]
            outputs = individual.forward(inputs)
            for i in range(len(outputs)):
                sum_error += (expected[i] - outputs[i])**2
        return -(sum_error / len(x))

    def run_epoch(self):
        child_population = []
        while len(child_population) <= self.pop_size:
            p1 = self.select_parent()
            p2 = self.select_parent()
            c1, c2 = self.crossover(p1, p2)
            self.mutate(c1)
            self.mutate(c2)
            self.population_fitness[c1] = self.calculate_fitness(c1)
            self.population_fitness[c2] = self.calculate_fitness(c2)
        
        # sort individuals by their fitness
        all_individuals = sorted(self.population_fitness.items(), key=lambda tuple: tuple[1], reverse=True)
        new_population = [all_individuals.pop() for i in range(self.pop_size)]
        # remove old networks from being tracked
        for individual, fitness in all_individuals:
            self.population_fitness.pop(individual)
        # set population to new population
        self.population = [individual for individual, fitness in new_population]

    def crossover(self, p1: MLFFNetwork, p2: MLFFNetwork):
        c1 = self.create_individual()
        c2 = self.create_individual()
        num_weights = p1.num_weights + p1.num_bias_weights
        # choose crossover point
        crossover_point = random.randint(0, num_weights)
        current_index = 0
        for layer in layer: # need to do index based
            for neuron in layer: # need ot index based
                weights = neuron.get_weights()
                for i in range(len(weights)):
                    if current_index == crossover_point:
                        pass
                    else:
                        pass
                    current_index += 1
