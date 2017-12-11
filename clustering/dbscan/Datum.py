import math

class Datum:

    def __init__(self, data):
        self.data = data
        self.point_type = None
        self.classification = None
        self.neighbors = []

    def __str__(self):
        return self.data

    def __repr__(self):
        return self.data

    def set_classification(self,new_class):
        self.classification = new_class

    def set_type(self, new_type):
        self.point_type = new_type

    def get_type(self):
        return self.point_type

    def get_classification(self):
        return self.classification

    def get_data(self):
        return self.data

    def add_neighbor(self, new_neighbor):
        self.neighbors.append(new_neighbor)

    def distance_from(self, target):
        if type(target) == Datum:
            squared_values = 0
            for local_feature,target_feature in zip(self.data, target.get_data()):
                squared_values += (local_feature - target_feature)**2
            return math.sqrt(squared_values)

        else:
            print("distance_from target must be of type Datum")
            raise TypeError