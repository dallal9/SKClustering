import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans



class metric:
    def __init__(self, X, Y, centroids):
        self.X = X
        self.Y = Y
        self.centroids = centroids
    
    def EConstant(self):
        km = KMeans(n_clusters=1).fit(self.X)
        
        e_constant = []
        labels_ = km.labels_
        cluster_k_const = [self.X[labels_== k] for k in range(km.n_clusters)] 
        
        sum_k = 0

        for i,k in enumerate(cluster_k_const):
            for r in [k]:
                df = pd.DataFrame(r)
                for index, row in df.iterrows():
                    sum_k += euclidean(row, km.cluster_centers_[0])
                    #sum_k += euclidean([row[0], row[1]], km.cluster_centers_[0])
                e_constant.append(sum_k)

        return e_constant[0]
        

    def IIndex(self):
        
        n_clusters = len(np.bincount(self.Y)) #Length of clusters
        
        cluster_k = [self.X[self.Y == k] for k in range(n_clusters)]
        e_constant = self.EConstant()

        #print("E_Constant: ", e_constant)
        
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

        e_ratio = e_constant/np.sum(ek)

        power = 0.5
        pair_distances = []

        for i in range(n_clusters):
            for j in range(n_clusters):
                if j != i:
                    pair_distances.append(euclidean(self.centroids[i], self.centroids[j]))
        
        index = ((1/float(n_clusters)) * e_ratio * np.max(pair_distances))**power

        return index