import os
import numpy as np
from scipy.spatial.distance import euclidean



class metric:
    def __init__(self, X, Y, centroids):
        self.X = X
        self.Y = Y
        self.centroids = centroids

    def IIndex(self):
        
        n_clusters = len(np.bincount(self.Y)) #Length of clusters
        
        cluster_k_const = [self.X[self.Y == k] for k in range(1)] 
        cluster_k = [self.X[self.Y == k] for k in range(n_clusters)]
        
        # Review partition matrix
        e_constant = [np.mean([euclidean(p, self.centroids[i]) for p in k]) for i, k in enumerate(cluster_k_const)]
        ek = [np.mean([euclidean(p, self.centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]
        e_ratio = e_constant/sum(ek)
        #print("EC", e_constant)
        #print("EK", sum(ek))
        
        power = 2
        pair_distances = []

        for i in range(n_clusters):
            for j in range(n_clusters):
                if j != i:
                    pair_distances.append(euclidean(self.centroids[i], self.centroids[j]))
        
        index = ((1/float(n_clusters)) * e_ratio[0] * np.max(pair_distances))**power

        return index