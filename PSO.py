from Particle import Particle
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
                self.gbest = particle
                self.gbest_fitness = fitness

    def run(self):
        for particle in self.particles:
            particle.update_velocity(self.gbest)
            particle.update_current_state(0,1)
        self.find_global_best()

x,y = sk_data.make_blobs(100,2,2)
test_data = [norm(x),y]
pso = PSO(2,2,1,2,.5,10,test_data)

plot.ion()
cmap = plot.cm.get_cmap('brg', 10)

for i in range(100):
    pso.run()
    for i in range(len(pso.particles)):
        for center in pso.particles[i].current_state:
            plot.scatter(center[0],center[1], c=cmap(i), s=3)
    plot.show()
    plot.pause(.1)
    plot.gcf().clear()

    print("Fitness: ",pso.gbest_fitness)
print("Best Fitness Achieved: ",pso.gbest_fitness)
print(pso.gbest)

#for particle in pso.particles:
#    print(particle)




