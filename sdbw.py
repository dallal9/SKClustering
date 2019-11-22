import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans



class sdbw:
    def __init__(self, X, Y, centroids):
        self.X = X
        self.Y = Y
        self.len = len(np.bincount(Y)) #Length of clusters
        self.centroids = centroids
   

    def sdbw_score(self):
        
        return 0