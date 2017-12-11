from clustering.dbscan.Datum import Datum
from sklearn import decomposition as dec
from dataset import DatasetLoader
from sklearn.preprocessing import normalize as norm
import matplotlib.pyplot as plot


class DBSCAN:

    def __init__(self, data, epsilon, minpts):
        self.data = self.raw_data_to_datums(data[0])
        self.classes = data[1]
        self.epsilon = epsilon
        self.minpts = minpts
        self.cores = []
        self.neighbors = []
        self.noise = []
        self.clusters_found = 0

    def raw_data_to_datums(self,data):
        list_of_datums = []
        for d in data:
            list_of_datums.append(Datum(d))
        return list_of_datums

    def classify_data(self):

        for datum in self.data:
            neighbor_count = 0
            neighbor_is_core = False

            for neighbor in self.data:
                if datum != neighbor and datum.distance_from(neighbor) <= self.epsilon:
                    neighbor_count += 1
                    datum.add_neighbor(neighbor)
                    neighbor_is_core = True if neighbor.get_type() == "core" else False

            if neighbor_count >= self.minpts:
                datum.set_type("core")
                self.cores.append(datum)
            elif neighbor_is_core:
                datum.set_type("neighbor")
            else:
                datum.set_type("noise")

        current_class = -1

        for core in self.cores:
            increment_class = True

            if core.get_classification() is None:
                for neighbor in core.neighbors:
                    if neighbor.get_classification() != None:
                        core.set_classification(neighbor.get_classification())

                        for neighbor in core.neighbors:
                            neighbor.set_classification(core.get_classification())

                        increment_class = False
                        break
            else:
                for neighbor in core.neighbors:
                    neighbor.set_classification(core.get_classification())
                current_class = core.get_classification()
                increment_class = False

            if increment_class:
                current_class += 1
                self.clusters_found = current_class if current_class > self.clusters_found else self.clusters_found
                core.set_classification(current_class)
                for neighbor in core.neighbors:
                    neighbor.set_classification(core.get_classification())

    def calculate_fitness(self):
        fitness = []
        for item in set(self.classes):
            percentages = [0 for i in range(self.clusters_found+1)]
            for cluster in range(self.clusters_found+1):
                cluster_count = 0
                class_count = 0
                for i in range(len(self.data)):
                    if self.classes[i] == item:
                        class_count += 1
                        if self.data[i].get_classification() == cluster:
                            cluster_count += 1
                percentages[cluster] = cluster_count/class_count
            fitness.append(percentages)
            print("Class: ", item, " Percentages: ", percentages)
        return fitness

    def plot_clusters(self):
        pca = dec.PCA(n_components=2)
        data_to_plot = [datum.get_data() for datum in self.data]
        class_labels = [datum.get_classification() for datum in self.data]
        pca_points = pca.fit_transform(data_to_plot)
        cmap = plot.cm.get_cmap('brg',len(set(class_labels)))
        for i in range(len(pca_points)):
            x,y = pca_points[i][0], pca_points[i][1]
            label = class_labels[i]
            if label != None:
                plot.scatter(x,y, c = cmap(label), s = 4)
            else:
                pass
        plot.show()

if __name__ == "__main__":
    data = DatasetLoader.load('ecoli')

    scanner = DBSCAN([data.X,data.CLASS_Y], .2, 4)  # You may want to disable to shuffling in dataset.Dataset
    scanner.classify_data()
    for d in scanner.data:
        print("Datum Type: " ,d.get_type(), '\tDatum Class', d.get_classification(), '\tDatum Neighbors: ', len(d.neighbors))
    scanner.calculate_fitness()
    print("Clusters Found: ", scanner.clusters_found+1)
    scanner.plot_clusters()