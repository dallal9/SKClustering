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
        max_distances = []

        #Review separation computation: MAX & MINs
        for i,k in enumerate(cluster):
            for r in [k]:
                df = pd.DataFrame(r)
                for a,b in enumerate(cluster):
                    for c in [b]:
                        if i != a:
                            df_2 = pd.DataFrame(c)
                            for row in df.itertuples():
                                for row_2 in df_2.itertuples():
                                    pair_distances.append(euclidean(row, row_2)) 
                    max_distances.append(np.max(pair_distances))
    
    return np.min(max_distances)


    def COPScore(self):
        
        n_clusters = len(np.bincount(self.Y)) #Length of clusters
        
        cluster_k = [self.X[self.Y == k] for k in range(n_clusters)]

        cluster_size = 0
        data_size = len(self.X)
        for i,k in enumerate(cluster_k):
            for r in [k]:
                df = pd.DataFrame(r)
                cluster_size += len(df)

        index = ((1/data_size) * cluster_size * (cohesion()/separation()))
        
        return index