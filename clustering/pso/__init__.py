from clustering.pso.particle import Particle
from sklearn.preprocessing import normalize as norm
from sklearn import datasets as sk_data
import matplotlib.pyplot as plot

class PSO:

    def __init__(self, centers, features, factor1, factor2, w, particles, data):
        self.data = data
        self.centers = centers
        self.features = features
        self.factor1, self.factor2 = factor1,factor2
        self.inertia = w
        self.particles = self.initialize_particles(particles)
        self.gbest = None
        self.gbest_fitness = None
        self.find_global_best()

    def initialize_particles(self,particles):
        return [Particle(self.centers, self.features, self.factor1, self.factor2, self.inertia) for i in range(particles)]

    def find_global_best(self):
        for particle in self.particles:
            particle.calculate_fitness(self.data)
            fitness = particle.fitness
            if self.gbest is None or fitness > self.gbest_fitness:
                self.gbest = particle.current_state
                self.gbest_fitness = fitness

    def run(self):
        for particle in self.particles:
            particle.update_velocity(self.gbest)
            particle.update_current_state(-10,10)
        self.find_global_best()


if __name__ == "__main__":
    nclusters=4

    x,y = sk_data.make_blobs(100,2,nclusters)
    test_data = [x,y]
    g_factor = 1.49
    p_factor = 1.49
    pso = PSO(nclusters,2,p_factor,g_factor,0.72,7,test_data)

    plot.ion()
    plot.show()

    cmap = plot.cm.get_cmap('brg', 10)

    for i in range(100):
        pso.run()
        for i in range(len(pso.particles)):
            for center in pso.particles[i].current_state:
                plot.scatter(center[0],center[1], c=cmap(i), s=3)
                plot.scatter([p[0] for p in x], [p[1] for p in x], color="blue")
                g_best = pso.gbest
                plot.scatter([p[0] for p in g_best], [p[1] for p in g_best], color="green")
        plot.pause(0.001)
        plot.gcf().clear()

        print("Fitness: ",pso.gbest_fitness)
    print("Best Fitness Achieved: ",pso.gbest_fitness)
    print(pso.gbest)

#for particle in pso.particles:
#    print(particle)