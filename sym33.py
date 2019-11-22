import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans



class sym_gd:
    def __init__(self, X, Y, centroids):
        self.X = X
        self.Y = Y
        self.len = len(np.bincount(Y)) #Length of clusters
        self.centroids = centroids
    
    def cohesion(self):
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
    
    def separation(self):
        
        cluster = [self.X[self.Y == k] for k in range(self.len)]
        euclid_distances = []
        ps_distance = 0
        
        for i,k in enumerate(cluster):
            for r in [k]:
                df = pd.DataFrame(r)
                for index, row in df.iterrows():
                    euclid_distances.append(euclidean(row, self.centroids[i])) 
        
    #    ps_distance = (0)/np.unique(euclid_distances) * 

    return 0


    def sym_gd_score(self):
       
        return self.cohesion()/self.separation()