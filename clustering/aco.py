# ACOCluster.py

import math
import numpy
import random
import numpy.random as nrand
import scipy.spatial.distance as scidist
import matplotlib.pylab as plt
from sklearn.preprocessing import normalize
import sklearn.datasets
from sklearn import metrics
import sklearn.neighbors as skneigh

from clustering import ClusteringAlgorithm
from heapq import heappush, heappop
from clustering.kmeans import euclidean
from collections import Counter
from dataset import DatasetLoader

class ACOCluster(ClusteringAlgorithm):

    def __init__(self,
                 x,
                 y,
                 h, w, ants, sims, n, c, freq, path="image"):
        self.x = x
        self.y = y
        self.height = h
        self.width = w
        self.num_ants = ants
        self.simulations = sims
        self.n = n
        self.c = c
        self.freq = freq

        # Initialize the grid
        self.grid = Grid(self.height, self.width, self.x, self.y, path)

    def run(self):
        """
        Method for running the ACO clustering algorithm
        """
        # Create the colony of ants
        colony = []

        for i in range(self.num_ants):
            # Create ant
            ant = Ant(random.randint(0, self.height - 1), random.randint(0, self.width - 1), self.grid)
            # Add ant to colony
            colony.append(ant)

        for i in range(self.simulations):
            for ant in colony:
                # Move ant in simulation
                ant.move(self.n, self.c)
            if i % self.freq == 0:
                # Output to track which step image is being saved
                print(i)
                s = "img" + str(i)
                self.grid.plot(s)
                # compute scores
                dead_ants = []
                predicted = []
                actual = []

                # Create an array of dead ants
                for i in range(self.width):
                    for j in range(self.height):
                        if self.grid.grid[i][j] is not None:
                            dead_ants.append((self.grid.grid[i][j], (i, j)))

                # Create heap and find k-closest neighbors to dead ant
                for ant, coordinate in dead_ants:
                    da_heap = []
                    for other_ant, other_coord in dead_ants:
                        if ant != other_ant:
                            da_heap.append((other_ant, euclidean(coordinate, other_coord)))
                    nearest_neightbors = sorted(da_heap,key=lambda x: x[1])[:self.n]
                    nearest_labels = [a[0].label for a in nearest_neightbors]
                    predicted_label = [label for label, count in Counter(nearest_labels).most_common(1)][0]
                    predicted.append(predicted_label)
                    actual.append(ant.label)

                # Calculate metrics and report to terminal
                homogeneity = metrics.homogeneity_score(actual, predicted)
                completeness = metrics.completeness_score(actual, predicted)
                print("%d: completeness=%f, homogeneity=%f" % (i, completeness, homogeneity))


class Grid:

    def __init__(self, height, width, x, y, path, rand=True):
        """
        Initialize a grid (2D array) of DeadAnt objects
        :param height: height of the grid
        :param width: width of the grid
        :param path: path of directory to store output
        :param matrix: 2D array of DeadAnts derived from input file
        """

        self.x = x
        self.y = y

        self.cmap = plt.cm.get_cmap('brg', 10)

        # Initialize an empty matrix of type DeadAnt
        self.dim = numpy.array([height, width])
        self.width = width
        self.height = height
        self.path = path
        # Store the dimensions of the grid
        self.grid = numpy.empty((height, width), dtype=DeadAnt)

        if rand:
            # This is used to fill the grid randomly
            self.random(0.25)
        # This makes the plot redraw
        plt.ion()
        plt.figure(figsize=(height, width))
        self.max_d = 0.001

    def random(self, spread):
        """
        Initialize grid randomly
        :param spread: the percentage of the grid to fill up
        """
        for i, item in enumerate(self.x):
            placed = False
            dead_ants = []
            while not placed:
                x, y = [random.randint(0, self.height - 1), random.randint(0, self.width - 1)]
                if self.grid[x][y] is None:
                    ant = DeadAnt(item, self.y[i])
                    self.grid[x][y] = ant
                    dead_ants.append(ant)
                    placed = True

            # precompute the distance from all ants to all other ants
            for ant in dead_ants:
                for other_ant in dead_ants:
                    if ant != other_ant:
                        ant.distance(other_ant)

    def matrix(self):
        """
        Reduces the grid (2D array of DeadAnt objects) to a matrix which can be visualized
        :return: matrix of the grid
        """
        matrix = numpy.empty((self.dim[0], self.dim[1]))
        matrix.fill(0)
        for y in range(self.dim[0]):
            for x in range(self.dim[1]):
                if self.grid[y][x] is not None:
                    matrix[y][x] = self.get_grid()[y][x].reduce()
        return matrix

    def plot(self, name="", save=True):
        """
        2D image of the grid
        :param name: name of the image
        :return:
        """
        plt.matshow(self.matrix(), cmap=self.cmap, fignum=0)
        # Option to save images
        if save:
            plt.savefig(self.path + name + '.png')
        plt.draw()

    def get_grid(self):
        return self.grid

    def calc_probability(self, d, y, x, n, c):
        """
        This calculates the probability of drop / pickup for any given DeadAnt, d
        :param d: the dead ant
        :param x: the x coord of the dead ant / ant carrying dead ant
        :param y: the y coord of the dead ant / ant carrying dead ant
        :param n: the size of the neighbourhood function
        :param c: constant for convergence control
        :return: the probability of
        """
        # Starting x and y locations
        y_start = y - n
        x_start = x - n
        total = 0.0
        # For each neighbour
        for i in range((n*2)+1):
            xi = (x_start + i) % self.dim[0]
            for j in range((n*2)+1):
                # If we are looking at a neighbour
                if j != x and i != y:
                    yj = (y_start + j) % self.dim[1]
                    # Get the neighbour, o
                    item = self.grid[xi][yj]
                    # Get the distance of o to x (to compare)
                    if item is not None:
                        s = d.distance(item)
                        total += s
        # Normalize the density by the largest distance found so far
        md = total / (math.pow((n*2)+1, 2) - 1)
        if md > self.max_d:
            self.max_d = md
        dens = total / (self.max_d * (math.pow((n*2)+1, 2) - 1))
        dens = max(min(dens, 1), 0)
        t = math.exp(-c * dens)
        probability = (1-t)/(1+t)
        return probability


class Ant:
    def __init__(self, y, x, grid):
        """
        Initialize ant object.
        :param y: the y coord
        :param x: the x coord
        :param grid: the grid
        """
        self.spot = numpy.array([y, x])
        self.carry = grid.get_grid()[y][x]
        self.grid = grid

    def move(self, n, c):
        """
        A recursive function for making ants move around the grid
        :param step_size: the size of each step
        """
        step_size = random.randint(1, 25)
        # Add some vector (-1,+1) * step_size to the ants location
        self.spot += nrand.randint(-1 * step_size, 1 * step_size, 2)
        # Mod the new location by the grid size to prevent overflow
        self.spot = numpy.mod(self.spot, self.grid.dim)
        # Get the object at that location on the grid
        item = self.grid.get_grid()[self.spot[0]][self.spot[1]]
        # If the cell is occupied, move again
        if item is not None:
            # If the ant is not carrying an object
            if self.carry is None:
                # Check if the ant picks up the object
                if self.pick_up_decide(n, c) >= random.random():
                    # Pick up the object and rem from grid
                    self.carry = item
                    self.grid.get_grid()[self.spot[0]][self.spot[1]] = None
                # If not then move
                else:
                    self.move(n, c)
            # If carrying an object then just move
            else:
                self.move(n, c)
        # If on an empty cell
        else:
            if self.carry is not None:
                # Check if the ant drops the object
                if self.drop_decide(n, c) >= random.random():
                    # Drop the object at the empty location
                    self.grid.get_grid()[self.spot[0]][self.spot[1]] = self.carry
                    self.carry = None

    def pick_up_decide(self, n, c):
        """
        Probability of picking up an object
        :param n: the neighborhood size
        :return: probability of picking up
        """
        ant = self.grid.get_grid()[self.spot[0]][self.spot[1]]
        return 1 - self.grid.calc_probability(ant, self.spot[0], self.spot[1], n, c)

    def drop_decide(self, n, c):
        """
        Probability of dropping an object
        :return: probability of dropping
        """
        ant = self.carry
        return self.grid.calc_probability(ant, self.spot[0], self.spot[1], n, c)


class DeadAnt:
    def __init__(self, data, label):
        """
        A DeadAnt object is a vector from row of input file
        :param data: the row vector
        """
        self.data = data
        self.distance_map = {}
        self.label = label

    def distance(self, deadAnt):
        """
        Returns the Euclidean distance between this dead ant and some other dead ant
        :param deadAnt: the other dead ant
        :return: Euclidean distance
        """
        if deadAnt in self.distance_map:
            return self.distance_map[deadAnt]
        else:
            distance = scidist.euclidean(self.data, deadAnt.data)
            self.distance_map[deadAnt] = distance
            return distance

    def reduce(self):
        """
        Convert the ant's label to a color
        """
        return self.label + 1


if __name__ == '__main__':
    # X, Y = sklearn.datasets.make_blobs(20, 2, 2, random_state=1)
    # X, Y = numpy.loadtxt('dataset_files/yeast.txt', usecols=range(7))

    ds = DatasetLoader.load("ecoli")
    X, Y = ds.X, ds.CLASS_Y
    length = ds.size
    dim = math.floor(math.sqrt(1*length))
    neighborhood = math.ceil(math.sqrt(dim))
    num_ants = int(0.1*ds.size)
    colony = ACOCluster(X, Y, dim, dim, num_ants, 100000, neighborhood, 10, 100, path="Output//ecoli")
    colony.run()
