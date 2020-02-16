# import os
import numpy as np
# import pandas as pd
# from scipy.spatial.distance import euclidean
# from sklearn.cluster import KMeans


class sdbw:
    def __init__(self, X, Y, centroids):
        self.X = X
        self.Y = np.array(Y)
        self.len = len(centroids)  # Number of clusters
        self.centroids = centroids

    def scatter(self):
        return self.cluster_variance() / (self.len * self.dataset_variance())

    def density(self, cluster_id):
        centroids = self.centroids
        sigma = self.get_stdev()
        density = 0
        if len(cluster_id) == 2:
            centroid = (centroids[cluster_id[0]] + centroids[cluster_id[1]]) / 2
        else:
            centroid = centroids[cluster_id[0]]

        for i in cluster_id:
            data_cluster = self.X[self.Y == i].values
            for j in data_cluster:
                if np.linalg.norm(j - centroid) <= sigma:
                    density += 1
        return density

    def dens_bw(self):
        cluster_density = []
        k = self.len
        result = 0
        for i in range(k):
            cluster_density.append(self.density(cluster_id=[i]))
        for i in range(k):
            for j in range(k):
                if i != j:
                    denominator = max(cluster_density[i], cluster_density[j])
                    if denominator != 0:
                        result += self.density([i, j]) / denominator
        return result / (k *(k - 1))

    def get_stdev(self):
        stdev = 0
        for cluster_id in range(self.len):
            std_i = np.std(self.X[self.Y == cluster_id],axis=0)
            stdev += np.sqrt(np.dot(std_i.T, std_i))  # euclidean norm ||x||
        return np.sqrt(stdev) / self.len

    def cluster_variance(self):

        total_variance = 0

        for cluster_id in range(self.len):
            sigma_i = np.std(self.X[self.Y == cluster_id], axis=0)
            total_variance += np.sqrt(np.dot(sigma_i.T, sigma_i))

        # the for loop above is equivalent to the below snippet

        # cluster = [self.X[self.Y == k] for k in range(self.len)]
        # total_variance = 0
        # for i, k in enumerate(cluster):
        #     for r in [k]:
        #         df = pd.DataFrame(r)
        #         cluster_len = len(df)
        #         variance = 0
        #         for index, row in df.iterrows():
        #             distance = euclidean(list(row), self.centroids[i])
        #             variance += (distance * distance)
        #         sigma = variance / cluster_len
        #         total_variance += (sigma * sigma)

        return total_variance

    def dataset_variance(self):

        dataset_std = np.std(self.X, axis=0)
        variance = np.sqrt(np.dot(dataset_std.T, dataset_std))

        # dataset_mean = np.mean(self.X)
        # cluster = [self.X[self.Y == k] for k in range(self.len)]
        # variance = 0
        #
        # for i, k in enumerate(cluster):
        #     for r in [k]:
        #         df = pd.DataFrame(r)
        #         for index, row in df.iterrows():
        #             distance = euclidean(list(row), dataset_mean)
        #             variance += (distance * distance)
        #
        # return variance / len(self.X)

        return variance

    def sdbw_score(self):
        return self.scatter() + self.dens_bw()

        # def density(self):
    #
    #     cluster = [self.X[self.Y == k] for k in range(self.len)]
    #     sigma = self.get_stdev()
    #     density = 0
    #
    #     for i, k in enumerate(cluster):
    #         for r in [k]:
    #             df = pd.DataFrame(r)
    #             for a, b in enumerate(cluster):
    #                 for c in [b]:
    #                     if i != a:
    #                         df_2 = pd.DataFrame(c)
    #                         density += self.rkk(sigma, df, df_2, self.centroids[i], self.centroids[a])
    #
    #     # density *= 1.0 * (self.len * (self.len - 1))
    #     density /= (self.len * (self.len - 1))
    #     return density


    # def gamma(self, sigma, a, b, centroid):
    #
    #     distance, density = 0, 0
    #
    #     for index, row in a.iterrows():
    #         distance = euclidean(list(row), centroid)
    #         if distance <= sigma:
    #             density += 1
    #
    #     for index, row in b.iterrows():
    #         distance = euclidean(list(row), centroid)
    #         if distance <= sigma:
    #             density += 1
    #     return density
    #
    # def rkk(self, sigma, a, b, c_1, c_2):
    #
    #     pair_centroid = (c_1 + c_2) / 2
    #     denom = np.max([self.gamma(sigma, a, b, c_1), self.gamma(sigma, a, b, c_2)])
    #     print(denom)
    #     res = self.gamma(sigma, a, b, pair_centroid) / denom
    #     # print(res)
    #     return res

    # def sdbw_score(self):
    #     return self.scatter() + self.density()
