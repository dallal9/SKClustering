import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans


class gdvar:
    def __init__(self, X, Y, centroids):
        self.X = X
        self.Y = Y
        self.len = len(np.bincount(Y)) #Length of clusters
        self.centroids = centroids
    
    def separation(self, type):

        cluster = [self.X[self.Y == k] for k in range(self.len)]
        distance = 0 # distance between points and centroid for all clusters

        for i,k in enumerate(cluster):
            for r in [k]:
                df = pd.DataFrame(r)
                for index, row in df.iterrows():
                    distance += euclidean(row, self.centroids[i])

        if type == 3 :
            return (2/self.len) * distance
    
    def cohesion(self, type):

       
        if type == 3:
            
            distance = 0
            cluster_size = 0
            cluster_k = [self.X[self.Y == k] for k in range(self.len)]

            for i,k in enumerate(cluster_k):
                for r in [k]:
                    df = pd.DataFrame(r)
                    cluster_size += len(df)
                    for a,b in enumerate(cluster_k):
                        for c in [b]:
                            if i != a:
                                df_2 = pd.DataFrame(c)
                                for row in df.itertuples():
                                    for row_2 in df_2.itertuples():
                                        distance += euclidean(row, row_2)
            return (1/cluster_size) * distance

        elif type == 4:
            pair_distances = []

            for i in range(self.len):
                for j in range(self.len):
                    if j != i:
                        pair_distances.append(euclidean(self.centroids[i], self.centroids[j]))
            return pair_distances


    def gD33(self):
        return self.cohesion(3)/self.separation(3)
    
    def gD43(self):
        return self.cohesion(4)/self.separation(3)