import os
import numpy as np
from scipy.spatial.distance import euclidean



class metric:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def IIndex(self):
        
        n_clusters = len(np.bincount(self.Y)) #Length of clusters
        n_points = len(self.X) #total number of points in dataset
        
        cluster_k = [self.X[self.Y == k] for k in range(n_clusters)]

        print("Clusters: ", n_clusters)
        
        centroids = [np.mean(k, axis = 0) for k in cluster_k]

        constant = [1]

        # Review partition matrix
        e_constant = np.sum(np.sum(euclidean(self.X[k], centroids[n]) for n in range(n_points)) for k in constant)
        ek = np.sum(np.sum(euclidean(self.X[k], centroids[n]) for n in range(n_points)) for k in range(n_clusters))
        e_ratio = e_constant/ek

        pair_distances = []

        for i in range(n_clusters):
            for j in range(n_clusters):
                if j != i:
                    pair_distances.append(euclidean(centroids[i], centroids[j]))


        return ((1/n_clusters) * e_ratio * np.max(pair_distances))