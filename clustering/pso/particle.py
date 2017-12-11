import random as rand
import math
import numpy as np
from sklearn.metrics import homogeneity_score as h_score, completeness_score as c_score

class Particle:

    def __init__(self, centers, features, f1, f2, w, min_bound, max_bound):
        self.current_state = np.random.uniform(min_bound, max_bound, (centers, features))#[np.random.uniform(features) for i in range(centers)]#[[rand.random() for f in range(features)] for i in range(centers)]
        self.velocity = np.random.uniform(0.1*min_bound, 0.1*max_bound, (centers, features)) # [np.random.rand(features) for i in range(centers)] #[[0 for f in range(features)] for i in range(centers)]
        self.personal_best = self.current_state
        self.factor1 = f1
        self.factor2 = f2
        self.inertia = w
        self.fitness = None

    def __str__(self):
        return 'State: ' + str(self.current_state) + '\nVelocity: ' + str(self.velocity)

    def update_velocity(self, gbest):
        c1 = np.random.uniform(0, self.factor1)
        c2 = np.random.uniform(0, self.factor2)

        for i in range(len(self.velocity)):
            self.velocity[i] = [(self.inertia*v) + (c1 * (p-c)) + (c2 * (g-c))
                              for v,p,g,c
                              in zip(self.velocity[i],self.personal_best[i],gbest[i],self.current_state[i])
                            ]

    def euclidian_distance(self, target, center):
        squared_values = 0
        for local_feature,target_feature in zip(center, target):
            squared_values += (local_feature - target_feature)**2

        return math.sqrt(squared_values)


    def update_current_state(self,min_bounds, max_bounds):
        for curr, vel in zip(self.current_state, self.velocity):
            for i in range(len(curr)):
                if curr[i]+vel[i] <= min_bounds:
                    curr[i] = min_bounds
                elif curr[i]+vel[i] >= max_bounds:
                    curr[i] = max_bounds
                else:
                    curr[i] = curr[i] + vel[i]

    def predict(self, dataset, with_distance=False):
        new_assignments = []
        for d in dataset:
            center_distances = []
            for index, center in enumerate(self.current_state):
                # append the output of the neuron and the neuron weights
                center_distances.append([index, self.euclidian_distance(center, d)])

            # find max by output and save the neuron's index
            min_center = min(center_distances, key=lambda item: item[1])
            new_assignments.append((min_center[0], min_center[1]))
        if with_distance:
            return new_assignments
        else:
            return [assignment[0] for assignment in new_assignments]

    # data expected as [[[],[],[],...],[classes]]
    def calculate_fitness(self, dataset):
        new_assignments = self.predict(dataset, with_distance=True)

        new_fitness = 0
        # calculate quantization error
        for index, center in enumerate(self.current_state):
            # create a list of all dataset items assigned to this cluster
            center_fitness = 0
            assignments = [assignment for assignment in new_assignments if assignment[0] == index]
            for assignment in assignments:
                center_fitness += assignment[1]
            if len(assignments) > 0:
                new_fitness += center_fitness / len(assignments)

        new_fitness /= len(self.current_state)
        # objective is to minimize the error
        new_fitness = -new_fitness

        if self.fitness is None or self.fitness < new_fitness:
            self.fitness = new_fitness
            self.personal_best = self.current_state
        return new_fitness







