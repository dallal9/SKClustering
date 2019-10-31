import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

class dunndex:
    def __init__(self, X, Y, centroids):
        self.X = X
        self.Y = Y
        self.centroids = centroids
    
    def dunn_index(self):
        
        n_clusters = len(np.bincount(self.Y)) #Length of clusters
        cluster_k = [self.X[self.Y == k] for k in range(n_clusters)]
        
        sum_k = 0
        big_deltas = []

        for i,k in enumerate(cluster_k):
            for r in [k]:
                df = pd.DataFrame(r)
                for index, row in df.iterrows():
                    big_deltas.append(euclidean([row[0], row[1]], self.centroids[i])) 
        
        deltas = []

        for i in range(n_clusters):
            for j in range(n_clusters):
                if j != i:
                    deltas.append(euclidean(self.centroids[i], self.centroids[j]))
        
        print("Min: ", np.min(deltas))
        print("Max: ", np.max(big_deltas))

        index = np.min(deltas)/np.max(big_deltas)

        return index
        
                