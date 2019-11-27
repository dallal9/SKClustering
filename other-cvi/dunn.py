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
                for row in df.itertuples():
                    for alt_row in df.itertuples():
                        if row.Index != alt_row.Index :
                            big_deltas.append(euclidean(list(row), list(alt_row))) 
                
        deltas = []

        for i,k in enumerate(cluster_k):
            for r in [k]:
                df = pd.DataFrame(r)
                for a,b in enumerate(cluster_k):
                    for c in [b]:
                        if i != a:
                            df_2 = pd.DataFrame(c)
                            for row in df.itertuples():
                                for row_2 in df_2.itertuples():
                                    deltas.append(euclidean(list(row), list(row_2))) 
       
       
        print("Cohesion: ", np.min(deltas))

        index = np.min(deltas)/np.max(big_deltas)

        return index
        
                