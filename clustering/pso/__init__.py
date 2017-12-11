from clustering import ClusteringAlgorithm
from clustering.pso.particle import Particle
from sklearn.preprocessing import normalize as norm
from sklearn import datasets as sk_data
from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plot

class PSO(ClusteringAlgorithm):

    def __init__(self,
                 num_features,
                 num_centers,
                 x,
                 y,
                 personal_factor=2,
                 global_factor=2,
                 w=0.5,
                 particles=10):
        self.X = x
        self.Y = y
        self.calculate_bounds()
        self.centers = num_centers
        self.features = num_features
        self.factor1, self.factor2 = personal_factor, global_factor
        self.inertia = w
        self.particles = self.initialize_particles(particles)
        self.gbest = None
        self.gbest_fitness = None
        self.find_global_best()

    def calculate_bounds(self):
        self.max_bound = np.amax(np.absolute(self.X))
        self.min_bound = -self.max_bound
        print("Computed bounds of (%f, %f)" % (self.min_bound, self.max_bound))

    def initialize_particles(self,particles):
        return [Particle(self.centers, self.features, self.factor1, self.factor2, self.inertia, self.min_bound, self.max_bound) for i in range(particles)]

    def find_global_best(self):
        for particle in self.particles:
            particle.calculate_fitness(self.X)
            fitness = particle.fitness
            if self.gbest is None or fitness > self.gbest_fitness:
                self.gbest = particle.current_state
                self.gbest_fitness = fitness
                self.gbest_particle = particle

    def run(self, max_epochs=1000):
        for epoch in range(max_epochs):
            for particle in self.particles:
                particle.update_velocity(self.gbest)
                particle.update_current_state(self.min_bound,self.max_bound)
            self.find_global_best()

            new_predictions = self.gbest_particle.predict(self.X)

            if epoch % 10 == 0:
                homogeneity = metrics.homogeneity_score(self.Y, new_predictions)
                completeness = metrics.completeness_score(self.Y, new_predictions)
                print("completeness=%f, homogeneity=%f, quantization_error=%f" % (completeness, homogeneity, self.gbest_fitness))

            yield epoch
#
# if __name__ == "__main__":
#     nclusters=2
#     n_features=2
#     x,y = sk_data.make_blobs(100,n_features,nclusters, cluster_std=1, random_state=1)
#     g_factor = 1.49
#     p_factor = 1.49
#     inertia=0.5
#     n_particles=7
#     pso = PSO(n_features,nclusters, p_factor,g_factor,inertia,n_particles, x, y)
#
#     plot.ion()
#     plot.show()
#
#     cmap = plot.cm.get_cmap('brg', 10)
#
#     for epoch in pso.run():
#        # pso.run()
#         for i in range(len(pso.particles)):
#             for center in pso.particles[i].current_state:
#                 plot.scatter(center[0],center[1], c=cmap(i), s=3)
#                 plot.scatter([p[0] for p in x], [p[1] for p in x], color="blue")
#                 g_best = pso.gbest
#                 plot.scatter([p[0] for p in g_best], [p[1] for p in g_best], color="green")
#         plot.pause(0.001)
#         plot.gcf().clear()
#
#        # print("Fitness: ",pso.gbest_fitness)
#     print("Best Fitness Achieved: ",pso.gbest_fitness)
#     print(pso.gbest)

#for particle in pso.particles:
#    print(particle)