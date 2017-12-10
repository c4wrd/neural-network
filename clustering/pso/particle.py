import random as rand
import math
from sklearn.metrics import homogeneity_score as h_score

class Particle:

    def __init__(self, centers, features, f1, f2, w):
        self.current_state = [[rand.random() for f in range(features)] for i in range(centers)]
        self.velocity = [[0 for f in range(features)] for i in range(centers)]
        self.personal_best = self.current_state
        self.factor1 = f1
        self.factor2 = f2
        self.inertia = w
        self.fitness = None

    def __str__(self):
        return 'State: ' + str(self.current_state) + '\nVelocity: ' + str(self.velocity)

    def update_velocity(self, gbest):
        rand1 = rand.random()
        rand2 = rand.random()

        for i in range(len(self.velocity)):
            self.velocity[i] = [self.inertia*v + self.factor1 * rand1 * (p-c) + self.factor2 * rand2 * (g-c)
                              for v,p,g,c
                              in zip(self.velocity[i],self.personal_best[i],gbest.current_state[i],self.current_state[i])
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

    # data expected as [[[],[],[],...],[classes]]
    def calculate_fitness(self, dataset):
        labels = [x for x in range(len(set(dataset[1])))]
        data = dataset[0]
        class_center_pairs = [[label,center] for label,center in zip(labels,self.current_state)]

        new_labels = []
        for d in data:
            for pair in class_center_pairs:
                dist = None
                curr_label = None
                if dist is None or dist > self.euclidian_distance(d,pair[1]):
                    dist = self.euclidian_distance(d,pair[1])
                    curr_label = pair[0]
            new_labels.append(curr_label)

        fitness = h_score(dataset[1], new_labels)

        if self.fitness is None or self.fitness < fitness:
            self.fitness = fitness
            self.personal_best = self.current_state
        return fitness







