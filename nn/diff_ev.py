import random

from dataset import Dataset
from nn.neural_network import NetworkShape
from nn.evolution import EvolutionaryStrategy, FitnessFunctions
from nn.mlff import MLFFNetwork


class DETrainer(EvolutionaryStrategy):
    def __init__(self, ps, pop_s, alpha, beta, dataset: Dataset, fitness_fn=FitnessFunctions.squared_error):
        self.population_size = ps
        self.pop_s = pop_s
        self.alpha = alpha
        self.beta = beta
        self.train = dataset.get_train()
        self.validation = dataset.get_validation()
        self.population_fitness = {}
        self.fitness_function = fitness_fn
        self.population = self.gen_pop()

    def big3(self,arr):
        return arr.index(max(arr))

    def gen_pop(self):
        population = []
        for p in range(self.population_size):
            new_indiv = MLFFNetwork(self.pop_s.num_inputs, self.pop_s.num_hidden_layers, self.pop_s.num_hidden_nodes, self.pop_s.num_outputs)
            self.population_fitness[new_indiv] = self.indiv_fitness(new_indiv)
            population.append(new_indiv)
        return population

    def crossover(self, p: MLFFNetwork, x1: MLFFNetwork, x2: MLFFNetwork, x3: MLFFNetwork):
        u = MLFFNetwork(self.pop_s.num_inputs, self.pop_s.num_hidden_layers, self.pop_s.num_hidden_nodes, self.pop_s.num_outputs)
        for pl,xl1,xl2,xl3,ul,layer_count in zip(p.layers, x1.layers,x2.layers,x3.layers, u.layers, range(self.pop_s.num_hidden_layers+1)):
            for pn, xn1, xn2, xn3, un, neuron_count in zip(pl, xl1, xl2, xl3, ul, range(len(xl1))):
                pw = pn.get_weights()
                xw1 = xn1.get_weights()
                xw2 = xn2.get_weights()
                xw3 = xn3.get_weights()
                for index in range(len(xw1)):
                    u_val = xw1[index] + self.beta*(xw2[index] - xw3[index])
                    if random.random() < self.alpha:
                        un.set_weight(index, pw[index])
                    else:
                        un.set_weight(index, u_val)
        return u

    def fitness(self, p: MLFFNetwork, u: MLFFNetwork):
        input, expected = zip(*[(row[0],row[1]) for row in self.data])
        error_u = 0
        error_p = 0
        for val,exp in zip(input,expected):
            actual_p = p.forward(val)
            actual_u = u.forward(val)
            for a,e in zip(actual_p,exp):
                error_p += (e-a)**2
            for a,e in zip(actual_u,exp):
                error_u += (e-a)**2
        return (p,error_p) if error_p < error_u else (u,error_u)

    def indiv_fitness(self, p):
        return self.fitness_function(self.train, p)

    def run_generation(self):
        new_population = []
        new_generation_fitness = {}

        for iteration in range(self.population_size):
            p = self.population[iteration]
            sample = random.sample(range(self.population_size),3)
            while iteration in sample:
                sample = random.sample(range(self.population_size),3)
            x1 = self.population[sample[0]]
            x2 = self.population[sample[1]]
            x3 = self.population[sample[2]]
            u = self.crossover(p,x1,x2,x3)
            fitness_u = self.indiv_fitness(u)
            if fitness_u > self.population_fitness[p]:
                # replace p with u
                new_generation_fitness[u] = fitness_u
                self.population_fitness[u] = fitness_u
                self.population_fitness.pop(p)

        sorted_population = sorted(self.population_fitness.items(), key=lambda pair: pair[1], reverse=True)

        # keep only the top 75%
        top75 = sorted_population[:int(self.population_size*.75)+1]
        for indiv, fitness in top75:
            new_population.append(indiv)
            new_generation_fitness[indiv] = fitness

        for i in range(int(self.population_size*.25)):
            new_indev = MLFFNetwork(self.pop_s.num_inputs, self.pop_s.num_hidden_layers, self.pop_s.num_hidden_nodes, self.pop_s.num_outputs)
            new_indiv_fitness = self.indiv_fitness(new_indev)
            new_population.append(new_indev)
            new_generation_fitness[new_indev] = new_indiv_fitness

        self.population = new_population
        self.population_fitness = new_generation_fitness

        max_fitness = self.fitness_function(self.train, self.get_fittest_individual())
        validation_fitness = self.fitness_function(self.validation, self.get_fittest_individual())
        return max_fitness, validation_fitness

    def get_fittest_individual(self):
        return max(self.population_fitness.items(), key=lambda pair: pair[1])[0]




