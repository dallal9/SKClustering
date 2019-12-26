import random 
from sklearn.cluster import KMeans, MeanShift, DBSCAN, \
    AffinityPropagation, SpectralClustering, AgglomerativeClustering, \
    OPTICS, Birch
from sklearn import metrics
import numpy as np
import pandas as pd
import csv
import pygmo as pg
from scipy.stats import spearmanr

class kmeans:
    def __init__(self):
        self.params=["n_clusters","algorithm","init","n_init"]

    def generate_pop(self,size=1):
        population=[]
        for i in range(size+1):
            n_clusters= random.choice(list(range(2,101)))
            init =  random.choice(['k-means++', 'random'])
            algorithm =  random.choice(['auto', 'full', 'elkan'])
            n_init = random.choice(list(range(10,25)))
            population.append(["kmeans",KMeans(n_clusters=n_clusters, algorithm = algorithm ,init=init, n_init=n_init)])
        return population

    def mutate(self,pop):
        p = self.generate_pop(size=1)[0]
        L = len(self.params)
        pos = random.randint(0,L-1)

        if pos<=3:
            pop[1].n_init= p[1].n_init
        if pos<=2:
            pop[1].init= p[1].init
        if pos<=1:
            pop[1].algorithm= p[1].algorithm
        if pos==0:
            pop[1].n_clusters= p[1].n_clusters

        return pop

    def cross_over(self,pop,pop2):
        p = pop2
        L = len(self.params)
        pos = random.randint(0,L-1)

        if pos<=3:
            pop[1].n_init= p[1].n_init
        if pos<=2:
            pop[1].init= p[1].init
        if pos<=1:
            pop[1].algorithm= p[1].algorithm
        if pos==0:
            pop[1].n_clusters= p[1].n_clusters

        return pop

class meanshift:
    def __init__(self):
        self.params=["cluster_all","bin_seeding","init","n_init"]

    def generate_pop(self,size=1):
        population=[]
        for i in range(size+1):
            cluster_all= random.choice([True,False])
            bin_seeding =  random.choice([True,False])
            bandwidth =  random.choice([None,1,2,3,4,5])
            max_iter = random.choice([200,300,400,500,600,700])
            population.append(["meanshift",MeanShift(cluster_all=cluster_all, bin_seeding = bin_seeding ,bandwidth=bandwidth, max_iter=max_iter)])
        return population

    def mutate(self,pop):
        p = self.generate_pop(size=1)[0]
        L = len(self.params)
        pos = random.randint(0,L-1)

        if pos<=3:
            pop[1].max_iter= p[1].max_iter
        if pos<=2:
            pop[1].bandwidth= p[1].bandwidth
        if pos<=1:
            pop[1].bin_seeding= p[1].bin_seeding
        if pos==0:
            pop[1].cluster_all= p[1].cluster_all

        return pop

    def cross_over(self,pop,pop2):
        p = pop2
        L = len(self.params)
        pos = random.randint(0,L-1)

        if pos<=3:
            pop[1].max_iter= p[1].max_iter
        if pos<=2:
            pop[1].bandwidth= p[1].bandwidth
        if pos<=1:
            pop[1].bin_seeding= p[1].bin_seeding
        if pos==0:
            pop[1].cluster_all= p[1].cluster_all

        return pop

class dbscan:
    def __init__(self):
        self.params=["eps","min_samples","metric","algorithm","leaf_size","p"]

    def generate_pop(self,size=1):
        population=[]
        for i in range(size+1):
            eps= random.choice([0.3,0.5,0.8,1,2,3,4,5,6,7,8,9,10])
            min_samples =  random.choice([5,10,15,20,30,50,100,150,200])
            metric =  random.choice( ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'])
            algorithm = random.choice(["auto","ball_tree","kd_tree","brute"])
            leaf_size=random.choice([5,10,15,20,25,30,40,50,100,150,200])
            p = leaf_size=random.choice([1,2,3])
            population.append(["dbscan",DBSCAN(eps=eps, metric = metric, min_samples=min_samples,algorithm=algorithm, leaf_size=leaf_size,p=p )])
        return population

    def mutate(self,pop):
        p = self.generate_pop(size=1)[0]
        L = len(self.params)
        pos = random.randint(0,L-1)
        if pos <=5:
            pop[1].eps= p[1].eps
        if pos <=4:
            pop[1].metric= p[1].metric
        if pos<=3:
            pop[1].min_samples= p[1].min_samples
        if pos<=2:
            pop[1].algorithm= p[1].algorithm
        if pos<=1:
            pop[1].leaf_size= p[1].leaf_size
        if pos==0:
            pop[1].p= p[1].p

        return pop

    def mutate(self,pop,pop2):
        p = pop2
        L = len(self.params)
        pos = random.randint(0,L-1)
        if pos <=5:
            pop[1].eps= p[1].eps
        if pos <=4:
            pop[1].metric= p[1].metric
        if pos<=3:
            pop[1].min_samples= p[1].min_samples
        if pos<=2:
            pop[1].algorithm= p[1].algorithm
        if pos<=1:
            pop[1].leaf_size= p[1].leaf_size
        if pos==0:
            pop[1].p= p[1].p

        return pop


class AffinPropagation:
    """
    Affinity Propagation
    """
    def __init__(self):
        self.params = ["damping", "max_iter", "affinity"]
        self.L = len(self.params)  # number of parameters

    @staticmethod
    def generate_pop(size=1):
        population = []
        for i in range(size+1):
            damping = random.uniform(0.5, 1)
            max_iter = random.randint(100, 300)
            affinity = random.choice(['euclidean', 'precomputed'])
            population.append(["affinity_propagation",
                               AffinityPropagation(damping=damping, max_iter=max_iter, affinity=affinity)])
        return population

    def mutate(self, pop):
        p = self.generate_pop(size=1)[0]
        L = self.L
        pos = random.randint(0, L-1)

        if pos <= 2:
            pop[1].damping = p[1].damping
        if pos <= 1:
            pop[1].max_iter = p[1].max_iter
        if pos == 0:
            pop[1].affinity = p[1].affinity

        return pop

    def cross_over(self, pop, pop2):
        p = pop2
        L = self.L
        pos = random.randint(0, L-1)

        if pos <= 2:
            pop[1].damping = p[1].damping
        if pos <= 1:
            pop[1].max_iter = p[1].max_iter
        if pos == 0:
            pop[1].affinity = p[1].affinity

        return pop


class SpectralCluster:
    """
    Spectral Clustering
    """
    def __init__(self):
        self.params = ["n_clusters", "eigen_solver",
                       "n_init", "gamma", "affinity"]
        self.L = len(self.params)  # number of parameters

    @staticmethod
    def generate_pop(size=1):
        population = []
        for i in range(size+1):
            n_clusters = random.randint(2, 100)
            eigen_solver = random.choice([None, 'arpack', 'lobpcg', 'amg'])
            n_init = random.randint(1, 20)
            gamma = random.uniform(0.5, 3)
            affinity = random.choice(['nearest_neighbors', 'rbf'])
            population.append(["spectral_clustering",
                               SpectralClustering(n_clusters=n_clusters,
                                                  eigen_solver=eigen_solver,
                                                  n_init=n_init,
                                                  gamma=gamma,
                                                  affinity=affinity)])
        return population

    def mutate(self, pop):
        p = self.generate_pop(size=1)[0]
        L = self.L
        pos = random.randint(0, L-1)

        if pos <= 4:
            pop[1].n_clusters = p[1].n_clusters
        if pos <= 3:
            pop[1].eigen_solver = p[1].eigen_solver
        if pos <= 2:
            pop[1].n_init = p[1].n_init
        if pos <= 1:
            pop[1].gamma = p[1].gamma
        if pos == 0:
            pop[1].affinity = p[1].affinity

        return pop

    def cross_over(self, pop, pop2):
        p = pop2
        L = self.L
        pos = random.randint(0, L-1)

        if pos <= 4:
            pop[1].n_clusters = p[1].n_clusters
        if pos <= 3:
            pop[1].eigen_solver = p[1].eigen_solver
        if pos <= 2:
            pop[1].n_init = p[1].n_init
        if pos <= 1:
            pop[1].gamma = p[1].gamma
        if pos == 0:
            pop[1].affinity = p[1].affinity

        return pop


class AgglomerativeCluster:
    """
    Agglomerative Clustering
    """
    def __init__(self):
        self.params = ["n_clusters", "linkage", "affinity"]
        self.L = len(self.params)  # number of parameters

    @staticmethod
    def generate_pop(size=1):
        population = []
        for i in range(size+1):
            n_clusters = random.randint(2, 100)
            linkage = random.choice(['ward', 'complete', 'average', 'single'])
            affinity = random.choice(['euclidean', 'l1', 'l2',
                                      'manhattan', 'cosine'])
            population.append(["agglomerative_clustering",
                               AgglomerativeClustering(n_clusters=n_clusters,
                                                       linkage=linkage,
                                                       affinity=affinity)])
        return population

    def mutate(self, pop):
        p = self.generate_pop(size=1)[0]
        L = self.L
        pos = random.randint(0, L-1)

        if pos <= 2:
            pop[1].n_clusters = p[1].n_clusters
        if pos <= 1:
            pop[1].linkage = p[1].linkage
        if pos == 0:
            pop[1].affinity = p[1].affinity

        return pop

    def cross_over(self, pop, pop2):
        p = pop2
        L = self.L
        pos = random.randint(0, L-1)

        if pos <= 2:
            pop[1].n_clusters = p[1].n_clusters
        if pos <= 1:
            pop[1].linkage = p[1].linkage
        if pos == 0:
            pop[1].affinity = p[1].affinity

        return pop


class Optics:
    """
    OPTICS Clustering
    """
    def __init__(self):
        self.params = ["min_samples", "max_eps", "metric",
                       "cluster_method", "algorithm"]
        self.L = len(self.params)  # number of parameters

    @staticmethod
    def generate_pop(size=1):
        population = []
        for i in range(size+1):
            min_samples = random.uniform(0, 1)
            max_eps = random.choice([np.inf, random.uniform(1, 100)])
            metric = random.choice(['cityblock', 'cosine', 'euclidean',
                                    'l1', 'l2', 'manhattan', 'braycurtis', 
                                    'canberra', 'chebyshev', 'correlation',
                                    'dice', 'hamming', 'jaccard', 'kulsinski',
                                    'mahalanobis', 'minkowski', 'rogerstanimoto',
                                    'russellrao', 'seuclidean', 'sokalmichener',
                                    'sokalsneath', 'sqeuclidean', 'yule'])
            cluster_method = random.choice(['xi', 'dbscan'])
            algorithm = random.choice(['auto', 'ball_tree', 'kd_tree', 'brute'])

            population.append(["OPTICS",
                               OPTICS(min_samples=min_samples, max_eps=max_eps, metric=metric,
                                      cluster_method=cluster_method, algorithm=algorithm)])
        return population

    def mutate(self, pop):
        p = self.generate_pop(size=1)[0]
        L = self.L
        pos = random.randint(0, L-1)

        if pos <= 4:
            pop[1].min_samples = p[1].min_samples
        if pos <= 3:
            pop[1].max_eps = p[1].max_eps
        if pos <= 2:
            pop[1].metric = p[1].metric
        if pos <= 1:
            pop[1].cluster_method = p[1].cluster_method
        if pos == 0:
            pop[1].algorithm = p[1].algorithm

        return pop

    def cross_over(self, pop, pop2):
        p = pop2
        L = self.L
        pos = random.randint(0, L-1)

        if pos <= 4:
            pop[1].min_samples = p[1].min_samples
        if pos <= 3:
            pop[1].max_eps = p[1].max_eps
        if pos <= 2:
            pop[1].metric = p[1].metric
        if pos <= 1:
            pop[1].cluster_method = p[1].cluster_method
        if pos == 0:
            pop[1].algorithm = p[1].algorithm

        return pop


class BirchClustering:
    """
    Birch Clustering
    """
    def __init__(self):
        self.params = ["threshold", "branching_factor",
                       "compute_labels", "copy"]
        self.L = len(self.params)  # number of parameters

    @staticmethod
    def generate_pop(size=1):
        population = []
        for i in range(size+1):
            threshold = random.uniform(0.2, 2)
            branching_factor = random.randint(1, 100)
            compute_labels = random.choice([True, False])
            copy = random.choice([True, False])

            population.append(["birch",
                               Birch(threshold=threshold, branching_factor=branching_factor,
                                     compute_labels=compute_labels, copy=copy)])
        return population

    def mutate(self, pop):
        p = self.generate_pop(size=1)[0]
        L = self.L
        pos = random.randint(0, L-1)

        if pos <= 3:
            pop[1].threshold = p[1].threshold
        if pos <= 2:
            pop[1].branching_factor = p[1].branching_factor
        if pos <= 1:
            pop[1].compute_labels = p[1].compute_labels
        if pos == 0:
            pop[1].copy = p[1].copy

        return pop

    def cross_over(self, pop, pop2):
        p = pop2
        L = self.L
        pos = random.randint(0, L-1)

        if pos <= 3:
            pop[1].threshold = p[1].threshold
        if pos <= 2:
            pop[1].branching_factor = p[1].branching_factor
        if pos <= 1:
            pop[1].compute_labels = p[1].compute_labels
        if pos == 0:
            pop[1].copy = p[1].copy

        return pop


class AutoClus:

    def __init__(self, dfile=""):
        data = pd.read_csv(dfile, header=None, na_values='?')
        self.data = data

        self.kmeans = kmeans()
        self.meanshift = meanshift()
        self.dbscan = dbscan()
        self.affinity_propagation = AffinPropagation()
        self.spectral_clustering = SpectralCluster()
        self.agglomerative_clustering = AgglomerativeCluster()
        self.optics = Optics()
        self.birch = BirchClustering()

        self.population = []

    def generate_pop(self, size=40):
        """

        :param size: size needs to be divisible by # of clustering algorithms
        :return:
        """
        nr_algorithms = 8  # number of clustering algorithms
        p = int(size / nr_algorithms)
        population = []
        population.extend(self.kmeans.generate_pop(p))
        population.extend(self.meanshift.generate_pop(p))
        population.extend(self.dbscan.generate_pop(p))
        population.extend(self.affinity_propagation.generate_pop(p))
        population.extend(self.spectral_clustering.generate_pop(p))
        population.extend(self.agglomerative_clustering.generate_pop(p))
        population.extend(self.optics.generate_pop(p))
        population.extend(self.birch.generate_pop(p))

        self.population = population
        return population

    def evaluate_pop(self):

        new_population = []
        vals12 = []
        vals3 = []
        indx = []
        for i in range(len(self.population)):
            Metrics = {}
            try:
                clustering = self.population[i][1].fit(self.data)
            except:
                continue

            if len(set(list(clustering.labels_))) == 1:
                continue
            try:
                sample_size = int(len(self.data)*0.1)
                if sample_size < 100:
                    sample_size = len(self.data)
                Metrics["davies_bouldin_score"] = metrics.davies_bouldin_score(self.data,  clustering.labels_)
                Metrics["silhouette_score"] = metrics.silhouette_score(self.data, clustering.labels_,
                                                                       metric='euclidean', sample_size=sample_size,
                                                                       random_state=0)
                Metrics["calinski_harabasz_score"] = metrics.calinski_harabasz_score(self.data,  clustering.labels_)
            except:
                continue
            indx.append(i)
            vals12.append([Metrics["davies_bouldin_score"], Metrics["silhouette_score"]])
            vals3.append(Metrics["calinski_harabasz_score"])

        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=vals12)
        ndf.reverse()

        print(ndf)

    def cross_over(self, pop1, pop2):
        if pop1[0] != pop2[0]:
            return None
        else:
            model = eval("self."+pop1[0])
            n_pop = model.mutate(pop1, pop2)
            return n_pop

    def mutation(self, population=[]):
        new_population = []
        for pop in population:
            model = eval("self."+pop[0])
            n_pop = model.mutate(pop)
            new_population.append(n_pop)

        return new_population

auto = AutoClus(dfile="test.csv")
auto.generate_pop(15)
auto.evaluate_pop()
