import numpy as np
from math import sqrt, pi
import networkx as nx
import matplotlib.pyplot as plt

class EuclidianNetwork:
    def __init__(self, N=100, delta=2.0):
        self.N = N
        self.delta = delta
        self.positions = None
        self.graph = None
        
        theta = 2 * pi / self.N
        angles = np.arange(0, 2 * pi, theta)
        self.positions = np.array([np.cos(angles), np.sin(angles)]).T

    def generate_graph(self, delta=None):
        if delta is None:
            delta = self.delta

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.N))

        #normalization for the probability
        norm = self.distance(self.positions[0], self.positions[1])

        #create links between adjacent nodes
        for i in range(self.N - 1):
            self.graph.add_edge(i, i + 1)
        self.graph.add_edge(self.N - 1, 0)

        #create random links
        created_nodes = 0
        while created_nodes < self.N:
            rand1 = np.random.randint(0, self.N)
            rand2 = np.random.randint(0, self.N)
            if rand1 == rand2:
                continue

            distance = self.distance(self.positions[rand1], self.positions[rand2])

            if np.random.rand() < (distance / norm) ** (-self.delta):
                self.graph.add_edge(rand1, rand2)
                created_nodes += 1

    def show(self):
        if self.graph is None:
            self.generate_graph()

        plt.figure(figsize=(6, 6))
        nx.draw_circular(self.graph, with_labels=False, node_size=0)
        plt.show()

    def get_graph(self):
        """Returns the graph created"""
        return self.graph
    
    def get_adjacency_matrix(self):
        """Returns the adjacency matrix of the graph"""
        return nx.adjacency_matrix(self.graph)
    
    def show_adjacency_matrix(self, invertY=True):
        """Show the adjacency matrix with matplotlib
        
        Parameter:
        inverder: bool
            set to true to have the (0,0) point to be 
            on the lowest-left corner (instead of upper left)
        """

        adj_matrix = nx.adjacency_matrix(self.graph).toarray()
        plt.imshow(adj_matrix, cmap='hot_r')
        
        if invertY:
            plt.gca().invert_yaxis()

        plt.xlabel('site i')
        plt.ylabel('site j')
        plt.show()

    @staticmethod
    def distance(point1, point2):
        return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)