import random
from nn.neural_network import NetworkShape
from nn.evolution import EvolutionStrategy
from nn.mlff import MLFFNetwork


class DETrainer(EvolutionStrategy):
    def __init__(self, ps, pop_s, alpha, beta, data: [[[],[]]]):
        self.population_size = ps
        self.pop_s = pop_s
        self.population = self.gen_pop()
        self.alpha = alpha
        self.beta = beta
        self.data = data
        self.population_fitness = {}

    def gen_pop(self):
        population = []
        for p in range(self.population_size):
            new_indiv = MLFFNetwork(self.pop_s.num_inputs, self.pop_s.num_hidden_layers, self.pop_s.num_hidden_nodes, self.pop_s.num_outputs)
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
                    un.set_weight(index, self.alpha*pw[index] + (1-self.alpha)*u_val)
        return u


    def fitness(self, p: MLFFNetwork, u: MLFFNetwork):
        input, expected = zip(*[(row[0],row[1]) for row in self.data])

        for val,exp in zip(input,expected):
            actual_p = p.forward(val)
            actual_u = u.forward(val)
            error_p = 0
            #print(str(actual_p) + " " + str(expected))
            for a,e in zip(actual_p,exp):
                error_p += (e-a)**2
            error_u = 0
            for a,e in zip(actual_u,exp):
                error_u += (e-a)**2
            return (p,error_p) if error_p < error_u else (u,error_u)

    def run_generation(self):
        new_population = []
        self.population_fitness = {}
        for iteration in range(self.population_size):
            sample = random.sample(range(self.population_size),4)
            p = self.population[sample[0]]
            x1 = self.population[sample[1]]
            x2 = self.population[sample[2]]
            x3 = self.population[sample[3]]
            u = self.crossover(p,x1,x2,x3)
            best = self.fitness(p,u)
            new_population.append(best[0])
            self.population_fitness[best[0]] = -(best[1])        # storing the negative of the squared error so that retrieivng max returns fittest indiv
        self.population = new_population

    def get_fittest_individual(self):
        return max(self.population_fitness.items(), key=lambda pair: pair[1])

from dataset import Datasets

shape = NetworkShape(7,2,15,3)

trainer = DETrainer(10, shape, .6, .5, Datasets.seeds())

for i in range(1000):
    trainer.run_generation()
    print(str(trainer.get_fittest_individual()))


