import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans



class cop:
    def __init__(self, X, Y, centroids):
        self.X = X
        self.Y = Y
        self.len = len(np.bincount(Y)) #Length of clusters
        self.centroids = centroids
    
    def cohesion(self):
        cluster = [self.X[self.Y == k] for k in range(self.len)]
        distance = 0

        for i,k in enumerate(cluster):
            for r in [k]:
                df = pd.DataFrame(r)
                for index, row in df.iterrows():
                    distance += euclidean(row, self.centroids[i])

        return distance * (1/self.len)
    
    def separation(self):
        
        cluster = [self.X[self.Y == k] for k in range(self.len)]
        pair_distances = []

        #Review separation computation: MAX & MINs
        for i,k in enumerate(cluster):
            for r in [k]:
                df = pd.DataFrame(r)
                for index, row in df.iterrows():
                    for alt_index, alt_row in df.iterrows():
                        if row != alt_row:
                            pair_distances.append(euclidean(row, alt_row))
    
    return (1/self.len)


    def COPScore(self):
        
        n_clusters = len(np.bincount(self.Y)) #Length of clusters
        
        cluster_k = [self.X[self.Y == k] for k in range(n_clusters)]

       
        
        ek = []
        sum = 0

        for i,k in enumerate(cluster_k):
            for r in [k]:
                df = pd.DataFrame(r)
                for index, row in df.iterrows():
                    sum += euclidean(row, self.centroids[i])
                    #sum += euclidean([row[0], row[1]], self.centroids[i])
                ek.append(sum)
                sum = 0

        e_ratio # = e_constant/np.sum(ek)

        power = 0.5
        pair_distances = []

        for i in range(n_clusters):
            for j in range(n_clusters):
                if j != i:
                    pair_distances.append(euclidean(self.centroids[i], self.centroids[j]))
        
        index = ((1/float(n_clusters)) * e_ratio * np.max(pair_distances))**power

        return index