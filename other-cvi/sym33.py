from __future__ import division
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from annoy import AnnoyIndex



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
        cluster_size = 0
        ps_distance = 0
        
        nrows, dim = self.X.shape
        for i,k in enumerate(cluster):
            for r in [k]:
                df = pd.DataFrame(r)
                cluster_size += len(df)
                ann = AnnoyIndex(dim, metric = 'euclidean')
                for index, row in df.iterrows():
                    ann.add_item(index, list(row))
                ann.build(10)
                for index, row in df.iterrows():
                    temp_nn = ann.get_nns_by_vector(list(row), 3, include_distances=True)
                    d_1 = temp_nn[1][1]
                    d_2 = temp_nn[1][2]
                    ps_distance += (np.mean([d_1, d_2]) * euclidean(list(row), self.centroids[i]))
    

        return (2/cluster_size) * ps_distance


    def sym_gd_score(self):
        return self.cohesion()/self.separation()