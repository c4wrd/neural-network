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

    def big3(self,arr):
        return arr.index(max(arr))

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
                #print('\n\n')
                #print(str(xw1) + '\n' + str(xw2) + '\n' + str(xw3) + '\n' +str(pw))
                for index in range(len(xw1)):
                    u_val = xw1[index] + self.beta*(xw2[index] - xw3[index])
                    if random.random() < self.alpha:
                        un.set_weight(index, pw[index])
                    else:
                        un.set_weight(index, u_val)
                        #un.set_weight(index, self.alpha*pw[index] + (1-self.alpha)*u_val)
        return u

    def fitness_class(self, p: MLFFNetwork, u: MLFFNetwork):
        input, expected = zip(*[(row[0],row[1]) for row in self.data])
        correct_u = 0
        correct_p = 0
        for val,exp in zip(input,expected):
            actual_p = p.forward(val)
            actual_u = u.forward(val)
            if self.big3(actual_p) == self.big3(exp):
                correct_p+=1
            if self.big3(actual_u) == self.big3(exp):
                correct_u+=1
        return (p, correct_p) if correct_p > correct_u else (u, correct_u)

    def fitness(self, p: MLFFNetwork, u: MLFFNetwork):
        input, expected = zip(*[(row[0],row[1]) for row in self.data])
        error_u = 0
        error_p = 0
        for val,exp in zip(input,expected):
            #print("VAL " + str(val))
            actual_p = p.forward(val)
            actual_u = u.forward(val)
            #print("ACTUAL P" + str(actual_p) + "\t ACTUAL U" + str(actual_u))
            for a,e in zip(actual_p,exp):
                error_p += (e-a)**2
            for a,e in zip(actual_u,exp):
                error_u += (e-a)**2
        return (p,error_p) if error_p < error_u else (u,error_u)

    def indiv_fitness(self, p):
        input, expected = zip(*[(row[0], row[1]) for row in self.data])
        error_p = 0
        for val, exp in zip(input, expected):
            actual_p = p.forward(val)
            for a, e in zip(actual_p, exp):
                error_p += (e-a)**2
        return (p, error_p)

    def run_generation(self):
        new_population = []
        copy_fitness = {}
        self.population_fitness = {}

        for iteration in range(self.population_size):
            p = self.population[iteration]
            sample = random.sample(range(self.population_size),3)
            while iteration in sample:
                sample = random.sample(range(self.population_size),3)
            x1 = self.population[sample[0]]
            x2 = self.population[sample[1]]
            x3 = self.population[sample[2]]
            u = self.crossover(p,x1,x2,x3)
            best = self.fitness(p,u)
            self.population_fitness[best[0]] = -best[1]        # storing the negative of the squared error so that retrieivng max returns fittest indiv

        for i in range(int(self.population_size*.75)+1):
            indiv = self.get_fittest_individual()
            self.population_fitness.pop(indiv[0])
            new_population.append(indiv[0])
            copy_fitness[indiv[0]] = indiv[1]

        for i in range(int(self.population_size*.25)):
            new_indiv = self.indiv_fitness(MLFFNetwork(self.pop_s.num_inputs, self.pop_s.num_hidden_layers, self.pop_s.num_hidden_nodes, self.pop_s.num_outputs))
            new_population.append(new_indiv[0])
            copy_fitness[new_indiv[0]] = -new_indiv[1]

        self.population = new_population
        self.population_fitness = copy_fitness

    def get_fittest_individual(self):
        return max(self.population_fitness.items(), key=lambda pair: pair[1])

from dataset import Datasets

if False:   # runs seeds
    shape = NetworkShape(7,1,3,3)
    trainer = DETrainer(10, shape, .8, .5, Datasets.seeds())
    for i in range(5000):
        trainer.run_generation()
        print(str(trainer.get_fittest_individual()))
    data = Datasets.seeds()
    input, expected = zip(*[(row[0], row[1]) for row in data])
    indiv = trainer.get_fittest_individual()[0]
    count = 0
    correct = 0
    for inp,exp in zip(input,expected):
        count += 1
        result = indiv.forward(inp)
        #print(str(inp))
        #print("RESULT: " + str(result) + "\tEXPECTED: " + str(exp))
        if trainer.big3(result) == trainer.big3(exp):
            print("CORRECT: " + str(result) + "," + str(exp))
            correct+=1
        else:
            print("Wrong: " + str(result) + "," + str(exp))
    print(str(100 * (correct / count)))

if True:            # runs appliances
    shape = NetworkShape(26, 1, 30, 1)
    trainer = DETrainer(30, shape, .8, .5, Datasets.appliances())
    for i in range(5000):
        trainer.run_generation()
        print(str(trainer.get_fittest_individual()))
    data = Datasets.appliances()
    input, expected = zip(*[(row[0], row[1]) for row in data])
    indiv = trainer.get_fittest_individual()[0]
    count = 0
    correct = 0
    for inp,exp in zip(input,expected):
        count += 1
        result = indiv.forward(inp)
        print("RESULT: " + str(result) + "\tEXPECTED: " + str(exp))



